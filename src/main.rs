use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;
use anyhow::Context;
use clap::Parser;
use crossterm::event;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::{Payload, Qdrant, QdrantError};
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder};
use serde::{Deserialize, Serialize};
use crate::Role::System;

const COMMAND_COLL_NAME: &'static str = "commands";

#[derive(Parser, Debug)]
/// A TUI For Chatting with Qmulo Local AI
struct Config {
    #[arg(short='H', long, default_value = "localhost:8000")]
    /// The hostname and port of the Qmulo LLM server
    llm_host: String,
    #[arg(short='c', long)]
    /// The directory where embedding models will be written to and read from on each start
    model_cache: String
}

struct ChatContext {
    endpoint: String,
    context: Vec<Message>,
    // Enable using the chat without Qdrant/embeddings if no commands are ever executed
    embedding_model: Option<TextEmbedding>,
    qclient: Option<Qdrant>,
    commands: BTreeMap<String, Command>,
}

impl ChatContext {
    fn new(config: &Config, sys_prompt: String) -> Result<Self, anyhow::Error> {
        let commands = BTreeMap::new();
        Ok(Self {
            endpoint: format!("http://{}/generate", config.llm_host),
            context: vec![Message{role: System, content: sys_prompt}],
            embedding_model: None,
            qclient: None,
            commands
        })
    }
    async fn initialize_commands(&mut self) -> Result<(), anyhow::Error> {
        self.embedding_model = Some(TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGELargeENV15Q)
                .with_show_download_progress(true)
                .with_cache_dir(PathBuf::from("~/Projects/llms/models/"))
        ).context("Failed to load local embedding model")?);
        self.qclient = Some(Qdrant::from_url("http://localhost:6334").build()
            .context("Failed to build Qdrant vector db client")?);
        match self.qclient.as_ref().unwrap().create_collection(
            CreateCollectionBuilder::new(COMMAND_COLL_NAME)
                .vectors_config(VectorParamsBuilder::new(1024, Distance::Dot))).await {
            Ok(_) => {},
            Err(e) => {
                match &e {
                    QdrantError::ResponseError{status} => {
                        if status.code() != tonic::Code::AlreadyExists {
                            return Err(e.into())
                        }
                    },
                    _ => return Err(e.into())
                }
            }
        }
        self.commands.insert("retry".into(), Command{
            id: "retry".into(),
            description: "delete the last assistant response and regenerate it again, or retry the last response".into(),
            f: Rc::new(Box::new(|ctx| {
                ctx.context.pop();
                ctx.send_context()
            })),
        });
        self.commands.insert("hint".into(), Command{
            id: "hint".into(),
            description: "add a message in the system role, further clarifying how the assistant should behave, or providing a suggestion for future responses.".into(),
            f: Rc::new(Box::new(|ctx| {
                println!("// Enter your hint below:");
                match read_message()? {
                    InputType::Prompt(prompt) => {
                        ctx.context.push(Message::system(prompt));
                        Ok(())
                    }
                    InputType::Command(_) => {
                        Err(anyhow::Error::msg("unable to process command input inside command shell"))
                    }
                }
            })),
        });
        self.commands.insert("system".into(), Command{
            id: "system".into(),
            description: "Overwrite the system prompt with a new one.".into(),
            f: Rc::new(Box::new(|ctx| {
                println!("// Enter the new system prompt below:");
                match read_message()? {
                    InputType::Prompt(prompt) => {
                        ctx.context.get_mut(0).unwrap().content = prompt;
                        Ok(())
                    }
                    InputType::Command(_) => {
                        Err(anyhow::Error::msg("unable to process command input inside command shell"))
                    }
                }
            }))
        });
        // get a token embedding for each command, build a vec of mappings
        let embeddedings = self.embedding_model.as_ref().unwrap().embed(self.commands.iter()
            .map(|(_, command)| format!("{}: {}", command.id, command.description)).collect(), None)?;
        let mut points: Vec<PointStruct> = Vec::new();
        for (idx, embedding) in embeddedings.into_iter().enumerate() {
            let (_, command) = self.commands.iter().nth(idx).unwrap();
            points.push(PointStruct::new(idx as u64 + 1, embedding, Payload::try_from(serde_json::to_value(command)?)?));
        }
        self.qclient.as_ref().unwrap().upsert_points(UpsertPointsBuilder::new(COMMAND_COLL_NAME, points)).await?;
        Ok(())
    }
    async fn run_command(&mut self, command: String) -> Result<(), anyhow::Error> {
        let mut embedding = self.embedding_model.as_ref().unwrap().embed(vec![format!("query: {}", command)], None)?;
        let first = embedding.pop().unwrap();
        let response = self.qclient.as_ref().unwrap().query(
            QueryPointsBuilder::new(COMMAND_COLL_NAME).query(first).with_payload(true)
        ).await?;
        let id = response.result[0].get("id");
        let command = match self.commands.get(id.as_str().unwrap()) {
            Some(command) => {
                println!("// Executing command '{}': {}", command.id, command.description);
                command.f.clone()
            }
            None => {
                return Err(anyhow::Error::msg(format!("Command not found: {}", command)));
            }
        };
        (*command)(self)
    }
    fn send_context(&mut self) -> Result<(), anyhow::Error> {
        let response = ureq::post(&self.endpoint)
            .set("content-type", "application/json")
            .send_json(&self.context)?
            .into_json::<ServerResponse>()?;
        self.context.push(Message::assistant(response.output.clone()));
        Ok(())
    }
    fn send_user_message(&mut self, message: String) -> Result<(), anyhow::Error> {
        self.context.push(Message::user(message));
        self.send_context()
    }
}

#[derive(Serialize)]
struct Command {
    id: String,
    description: String,
    #[serde(skip)]
    f: Rc<Box<dyn Fn(&mut ChatContext) -> Result<(), anyhow::Error> + 'static>>
}

enum InputType {
    Prompt(String),
    Command(String),
}

impl InputType {
    fn into_string(self) -> String {
        match self {
            InputType::Prompt(prompt) => prompt,
            InputType::Command(cmd) => cmd
        }
    }
}

fn read_message() -> io::Result<InputType> {
    let mut line = String::new();
    while let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
        match code {
            KeyCode::Enter => {
                if modifiers.contains(KeyModifiers::ALT) {
                    break;
                }
                line.push('\n');
            }
            KeyCode::Char(c) => {
                line.push(c);
            }
            _ => {
                println!("Unknown key {:?}", code);
            }
        }
    }
    if line.starts_with('/') {
        Ok(InputType::Command(line[1..].to_string()))
    } else {
        Ok(InputType::Prompt(line))
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: Role,
    content: String,
}

impl Message {
    fn user(content: String) -> Self {
        Self{
            role: Role::User,
            content,
        }
    }
    fn system(content: String) -> Self {
        Self{
            role: Role::System,
            content,
        }
    }
    fn assistant(content: String) -> Self {
        Self{
            role: Role::Assistant,
            content,
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
enum Role {
    User,
    System,
    Assistant
}

#[derive(Deserialize, Serialize, Debug)]
struct ServerResponse {
    output: String,
    time: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    println!("Enter the system prompt for this session below: ");
    let sys_prompt = read_message()?;
    let mut ctx = ChatContext::new(&config, sys_prompt.into_string())?;
    ctx.initialize_commands().await?;
    println!("Now you can start chatting. Further responses will be from the assistant\n--------");
    loop {
        let prompt = read_message()?;
        match prompt {
            InputType::Prompt(prompt) => {
                ctx.send_user_message(prompt)?;
                println!("{}", ctx.context[ctx.context.len() - 1].content);
            }
            InputType::Command(cmd) => {
                if let Err(err) = ctx.run_command(cmd).await {
                    println!("// Command error: {}", err);
                }
            }
        };
    }
}
