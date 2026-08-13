//! Smoke test: verify OpenAIDriver.complete() with deepseek-chat produces
//! full-length output (not capped at ~64 tokens by the bounded-channel bug).
//!
//! Run: DEEPSEEK_API_KEY=... cargo run -p openfang-runtime --example test_deepseek

use openfang_runtime::drivers::openai::OpenAIDriver;
use openfang_runtime::llm_driver::{CompletionRequest, LlmDriver};
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY required");
    let driver = OpenAIDriver::new(api_key, "https://api.deepseek.com/v1".to_string());

    let prompt = "用中文写一段 800 字左右的故事，关于一个修仙少女误入现代都市的经历。不要列提纲，直接开始写。";

    let request = CompletionRequest {
        model: "deepseek-chat".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: prompt.to_string(),
            }]),
        }],
        tools: vec![],
        max_tokens: 1600,
        temperature: 0.0,
        system: None,
        thinking: None,
    };

    println!("Calling driver.complete() with deepseek-chat, max_tokens=1600...");
    let start = std::time::Instant::now();
    let resp = driver.complete(request).await?;
    let elapsed = start.elapsed();

    let text = resp.text();
    println!("=== elapsed: {:?}", elapsed);
    println!("=== stop_reason: {:?}", resp.stop_reason);
    println!(
        "=== usage: input={} output={}",
        resp.usage.input_tokens, resp.usage.output_tokens
    );
    println!("=== output chars: {}", text.chars().count());
    println!("--- first 300 chars ---");
    println!("{}", text.chars().take(300).collect::<String>());
    println!("--- last 200 chars ---");
    let last: String = text
        .chars()
        .rev()
        .take(200)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    println!("{}", last);

    if resp.usage.output_tokens <= 100 {
        eprintln!(
            "\n*** BUG STILL PRESENT: output_tokens={} (expected > 500) ***",
            resp.usage.output_tokens
        );
        std::process::exit(1);
    } else {
        println!(
            "\n*** OK: output_tokens={} > 100 — fix confirmed ***",
            resp.usage.output_tokens
        );
    }
    Ok(())
}
