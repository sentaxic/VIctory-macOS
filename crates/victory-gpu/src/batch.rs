//! The clamps of many indirect executions in one dispatch
//! (`crate::indirect`, "batches"): the table the batch's program reads.
//! Plain Rust, so that its tests and the replayer's use of it run
//! everywhere.

/// Words of a clamp in a batch's table (the program's `ENTRY`).
pub(crate) const ENTRY: usize = 10;
/// Buffers a batch's clamps read their records and counts from.
pub(crate) const BATCH_SOURCES: usize = 4;

/// One clamp of a batch: what one dispatch of the clamp would be given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
    pub(crate) flags: u32,
    /// The source buffer (of [`BATCH_SOURCES`]) and byte offset of the
    /// records, and of the count.
    pub(crate) from: (u32, u32),
    pub(crate) count: (u32, u32),
    /// The byte offset of the clamped records in the destination.
    pub(crate) to: u32,
}

impl Entry {
    /// Its words in the table.
    pub(crate) fn words(&self) -> [u32; ENTRY] {
        [
            self.kind,
            self.stride,
            self.records,
            self.flags,
            self.from.0,
            self.from.1,
            self.count.0,
            self.count.1,
            self.to,
            0,
        ]
    }
}
