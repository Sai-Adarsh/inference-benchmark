use std::io::{self, BufRead};

const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

pub struct Event {
    pub kind: String,
    pub data: String,
}

pub struct Events<R> {
    reader: R,
    first_line: bool,
    after_cr: bool,
    line: Vec<u8>,
}

impl<R: BufRead> Events<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, first_line: true, after_cr: false, line: Vec::new() }
    }

    // SSE permits LF, CRLF, and CR, including a CRLF pair split across reads.
    fn read_line(&mut self) -> io::Result<bool> {
        self.line.clear();
        if self.after_cr {
            self.after_cr = false;
            if self.reader.fill_buf()?.first() == Some(&b'\n') {
                self.reader.consume(1);
            }
        }
        loop {
            let chunk = self.reader.fill_buf()?;
            if chunk.is_empty() {
                return Ok(!self.line.is_empty());
            }
            let end = chunk.iter().position(|byte| matches!(byte, b'\n' | b'\r'));
            let length = end.unwrap_or(chunk.len());
            if length > MAX_PAYLOAD_BYTES - self.line.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "SSE line exceeds 1 MiB"));
            }
            self.line.extend_from_slice(&chunk[..length]);
            if let Some(end) = end {
                self.after_cr = chunk[end] == b'\r';
                self.reader.consume(end + 1);
                return Ok(true);
            }
            self.reader.consume(length);
        }
    }

    pub fn next(&mut self) -> io::Result<Option<Event>> {
        let mut kind = String::new();
        let mut data = String::new();
        while self.read_line()? {
            let mut line = std::str::from_utf8(&self.line)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            if self.first_line {
                self.first_line = false;
                line = line.strip_prefix('\u{feff}').unwrap_or(line);
            }
            if line.is_empty() {
                if data.pop().is_some() {
                    return Ok(Some(Event { kind, data }));
                }
                kind.clear();
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => {
                    kind.clear();
                    kind.push_str(value);
                }
                "data" => {
                    if value.len() >= MAX_PAYLOAD_BYTES - data.len() {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "SSE event exceeds 1 MiB"));
                    }
                    data.push_str(value);
                    data.push('\n');
                }
                _ => {} // Includes comments, event IDs, and retry hints.
            }
        }
        // An event without the final blank line is incomplete.
        Ok(None)
    }
}
