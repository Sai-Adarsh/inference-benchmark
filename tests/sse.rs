#[path = "../src/sse.rs"]
mod sse;

use std::io::{BufReader, ErrorKind};

#[test]
fn split_crlf_bom_comments_and_multiline_data() {
    let input = "\u{feff}: keepalive\revent: replaced\revent: partial\r\nretry: 10\nid: 4\ndata:  first\r\ndata: second\r\n\r\nevent: forgotten\n\ndata\n\n";
    // A one-byte buffer forces each CRLF and UTF-8 sequence across read boundaries.
    let mut events = sse::Events::new(BufReader::with_capacity(1, input.as_bytes()));
    let event = events.next().unwrap().unwrap();
    assert_eq!(event.kind, "partial");
    assert_eq!(event.data, " first\nsecond");
    let event = events.next().unwrap().unwrap();
    assert_eq!(event.kind, "");
    assert_eq!(event.data, "");
    assert!(events.next().unwrap().is_none());
}

#[test]
fn does_not_dispatch_a_final_unterminated_event() {
    for input in [
        b"data: partial".as_slice(),
        b"data: partial\n",
        b"data: partial\r",
    ] {
        assert!(sse::Events::new(input).next().unwrap().is_none());
    }
}

#[test]
fn validates_utf8_and_bounds_both_lines_and_assembled_events() {
    const MAX: usize = 1024 * 1024;
    let line = format!("data: {}\n\n", "x".repeat(MAX - 6));
    assert_eq!(
        sse::Events::new(line.as_bytes())
            .next()
            .unwrap()
            .unwrap()
            .data
            .len(),
        MAX - 6
    );
    let combined = format!(
        "data: {}\ndata: {}\n\n",
        "x".repeat(MAX / 2),
        "x".repeat(MAX / 2)
    );
    let cases: [(&[u8], &str); 3] = [
        (b"data: \xff\n\n", "utf-8"),
        (&vec![b'x'; MAX + 1], "SSE line exceeds"),
        (combined.as_bytes(), "SSE event exceeds"),
    ];
    for (input, message) in cases {
        let error = sse::Events::new(BufReader::new(input))
            .next()
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains(message), "{error}");
    }
}
