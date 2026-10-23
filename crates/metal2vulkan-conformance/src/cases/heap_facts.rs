//! What an automatic or placement `MTLHeap` answers about **itself**: the eight properties it
//! declares, its `device`, its label, `maxAvailableSizeWithAlignment:`,
//! `newBufferWithLength:options:` and `setPurgeableState:`.
//!
//! Before this family the suite called six of `MTLHeap`'s seventeen items, all through
//! [`crate::cases::sparse_heaps`] and the heap-placed texture families: `size`, `usedSize`, `type`,
//! `newTextureWithDescriptor:` and the two offset forms. Everything a heap says about **what kind
//! of memory it is** — its storage mode, its CPU cache mode, the hazard tracking mode Metal
//! resolved its descriptor's to, the option word it packs those into, its allocated size, the
//! longest buffer its suballocator will still place, and what a purgeable-state call answers — was
//! never compared against Metal at all.
//!
//! # The cases, and what each one would catch
//!
//! - **`heap-own-getters`**: twelve heaps — both types, both storage modes a heap takes, both CPU
//!   cache modes and all three hazard tracking modes — each one's ten answers. The row that
//!   matters most is `hazardTrackingMode` of a heap whose descriptor said `Default`: the runtime
//!   models it as **untracked**, which is the opposite of what a resource outside a heap resolves
//!   to, and nothing in this tree had asked Metal. A release where Apple resolves it the other way
//!   changes which barriers a consumer needs, and this is the case that would say so.
//! - **`heap-available-sizes`**: `maxAvailableSizeWithAlignment:` at seven alignments on an
//!   automatic heap, read **three times** — fresh, after one buffer is placed, and after a second
//!   — with the heap's `usedSize` and `currentAllocatedSize` beside each reading, and the same
//!   sweep on a placement heap. This is where Metal's own suballocator is the only possible
//!   oracle: the runtime models the longest free run at a multiple of the alignment, 0 for a
//!   placement heap whatever the alignment, and both are claims about Apple's allocator rather than
//!   about anything a descriptor states.
//! - **`heap-placed-buffer-sizes`**: `newBufferWithLength:options:` of an automatic heap at
//!   lengths either side of the 256-byte placement grain, each with the heap's three size figures
//!   and the placed buffer's own `heapOffset` after it. What `usedSize` counts is Metal's to
//!   settle and nothing in this tree had asked: the runtime sums each buffer's **length** rather
//!   than the padded extent its suballocator really took, and this chip answers the same — 1, 256,
//!   512, 769 and 4865 bytes as the five lengths are placed. `heapOffset` is the other half, and
//!   it is 0 at every row: an automatic heap's buffer reports no offset, so the suballocator's own
//!   placement is not observable through `MTLResource` at all. That is a hole this family names
//!   rather than one it closes.
//! - **`heap-purgeable-states`**: the `Volatile`, `Empty`, `KeepCurrent` chain on a shared and on a
//!   private automatic heap. Every answer is the state **before** its call, so the three readings
//!   are a transcript of Metal's own transition rule and none of them is the argument handed in.
//! - **`heap-labels`**: a heap's `label` before anything set one, then through
//!   [`crate::heap_query::LABEL_CHAIN`]. The first reading and the empty-string readings are the
//!   measured ones: whether Metal tells `nil` from `@""` is a fact about its own storage of the
//!   property, and the runtime answers `Some("")` for the second, which would be a divergence if
//!   Metal answers nil.
//!
//! # The positive control, and why this family needs one
//!
//! A getter sweep over an object the case itself configured is exactly the shape that can pass
//! while measuring nothing: read `storageMode` back off a heap whose descriptor named it and the
//! answer was never in doubt. So `heap-own-getters` asserts, in its own test and with no device at
//! all, that **every heap of its table states a `Default` hazard tracking mode somewhere and a
//! non-default one somewhere**, that both CPU cache modes appear, and that both heap types do —
//! and the case records `size` beside the size its descriptor asked for, where Metal's rounding is
//! a number nothing in the case wrote. A run that read a heap nobody made would answer the
//! unmade-handle refusal at every site rather than a size of 16384.
//!
//! # Why every case here is safe
//!
//! Nothing is encoded, dispatched or committed and no case reaches a queue: a heap is made, its
//! getters are read, at most two buffers are placed in it, and the purgeable-state call is asked
//! last of whatever asks it. The largest heap of the family is 1 MiB.
//!
//! The hazards a heap descriptor carries are named in [`crate::heap_query::HeapSpec`]'s own guard —
//! a sparse type, managed storage, a size of 0 — and this module's own tests hold every table to it with no
//! device. `maxAvailableSizeWithAlignment:` is asked only at powers of two, for the reason
//! [`crate::heap_query`] gives.

use crate::case::{Abort, Case, Handle, Input, MetalSource, Side};
use crate::heap_query::{
    purgeable_word, HeapBufferSpec, HeapSpec, QueriedResource, CACHE_MODES, HAZARD_MODES,
    LABEL_CHAIN, PURGEABLE_CHAIN,
};
use crate::texture_query::Storage;
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan_runtime::HeapType;

/// The case names, as the report and the recordings spell them.
pub const OWN_GETTERS: &str = "heap-own-getters";
pub const AVAILABLE_SIZES: &str = "heap-available-sizes";
pub const PLACED_BUFFERS: &str = "heap-placed-buffer-sizes";
pub const PURGEABLE: &str = "heap-purgeable-states";
pub const LABELS: &str = "heap-labels";

/// The alignments the available-size sweep asks about: 1, the 256-byte placement grain either
/// side, and up to a 64-KiB page.
///
/// Every one a power of two, which is what [`crate::heap_query`] says the sweep is kept inside: an
/// alignment of 0 is a shape nothing in this tree says Metal takes, and any other non-power-of-two
/// is one the runtime refuses `Invalid`.
const ALIGNMENTS: [u64; 7] = [1, 2, 16, 128, 256, 4096, 1 << 16];

/// The storage modes a heap of this family takes.
///
/// Two, not three: Metal's validation stops a managed heap, which
/// [`HeapSpec::is_well_formed`](crate::heap_query::HeapSpec::is_well_formed) refuses outright.
const BOTH_STORAGE: [Storage; 2] = [Storage::Shared, Storage::Private];

/// The heap size every case of the family asks for: 1 MiB, as
/// [`crate::cases::sparse_heaps`]' is, so a reader can put the two families' figures beside each
/// other.
const SIZE: u64 = 1 << 20;

/// The lengths `heap-placed-buffer-sizes` places, in order: either side of the 256-byte grain,
/// then two whole multiples.
///
/// A length of 0 is not here, for the reason [`crate::device_queries`] gives for its own sweep:
/// nothing in this tree says Metal's automatic suballocator takes one. The placement form does
/// take 0 — the runtime's documentation names a Metal probe for it — and that is
/// [`crate::cases::resource_facts`]' row, where the offset is the case's own.
const PLACED_LENGTHS: [u64; 5] = [1, 255, 256, 257, 4096];

/// What one case of the family asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ask {
    /// Every getter of every heap of the table.
    OwnGetters,
    /// `maxAvailableSizeWithAlignment:` at [`ALIGNMENTS`], three times over.
    AvailableSizes,
    /// `newBufferWithLength:options:` at [`PLACED_LENGTHS`], the heap's figures after each.
    PlacedBuffers,
    /// The purgeable-state chain.
    Purgeable,
    /// The label chain.
    Labels,
}

/// A case over one table of heaps.
pub struct HeapFactsCase {
    name: &'static str,
    ask: Ask,
    heaps: Vec<HeapSpec>,
}

/// Every case of the family.
pub fn all() -> Vec<HeapFactsCase> {
    vec![
        own_getters(),
        available_sizes(),
        placed_buffers(),
        purgeable_states(),
        labels(),
    ]
}

/// Both heap types in both storage modes, then one storage mode across both CPU cache modes and
/// all three hazard tracking modes.
///
/// The second half is what makes the first half mean something: a table that only ever stated
/// Metal's defaults would record one answer for `cpuCacheMode` and one for `hazardTrackingMode`,
/// and a reading that ignored the descriptor entirely would look exactly the same.
fn own_getters() -> HeapFactsCase {
    let mut heaps = Vec::new();
    for heap_type in [HeapType::Automatic, HeapType::Placement] {
        for storage in BOTH_STORAGE {
            heaps.push(of_type(heap_type, SIZE, storage));
        }
    }
    for heap_type in [HeapType::Automatic, HeapType::Placement] {
        for cache in CACHE_MODES {
            for hazard in HAZARD_MODES {
                // One storage mode here, because the field under test is the other two: shared,
                // so the CPU cache mode is a mode the memory could honestly have.
                let heap = of_type(heap_type, SIZE, Storage::Shared)
                    .cached(cache)
                    .tracked(hazard);
                if !heaps.contains(&heap) {
                    heaps.push(heap);
                }
            }
        }
    }
    HeapFactsCase {
        name: OWN_GETTERS,
        ask: Ask::OwnGetters,
        heaps,
    }
}

/// One automatic heap and one placement heap in both storage modes: the available-size sweep reads
/// each three times, so four heaps is twelve readings of seven alignments.
fn available_sizes() -> HeapFactsCase {
    let mut heaps = Vec::new();
    for heap_type in [HeapType::Automatic, HeapType::Placement] {
        for storage in BOTH_STORAGE {
            heaps.push(of_type(heap_type, SIZE, storage));
        }
    }
    HeapFactsCase {
        name: AVAILABLE_SIZES,
        ask: Ask::AvailableSizes,
        heaps,
    }
}

/// The automatic heaps whose suballocator the placed-buffer sweep reads.
///
/// Automatic alone: `newBufferWithLength:options:` without an offset is the automatic heap's call,
/// and a placement heap refuses it in both APIs — which is a refusal case rather than a
/// measurement of where Metal places a buffer.
fn placed_buffers() -> HeapFactsCase {
    HeapFactsCase {
        name: PLACED_BUFFERS,
        ask: Ask::PlacedBuffers,
        heaps: BOTH_STORAGE
            .into_iter()
            .map(|storage| of_type(HeapType::Automatic, SIZE, storage))
            .collect(),
    }
}

/// An automatic and a placement heap in both storage modes, for the purgeable chain.
///
/// The storage mode is in it because the runtime's model parts on it: emptying a shared heap may
/// discard its bytes where emptying a private one does not, and whether the **answers** part too
/// is what this records.
fn purgeable_states() -> HeapFactsCase {
    let mut heaps = Vec::new();
    for heap_type in [HeapType::Automatic, HeapType::Placement] {
        for storage in BOTH_STORAGE {
            heaps.push(of_type(heap_type, SIZE, storage));
        }
    }
    HeapFactsCase {
        name: PURGEABLE,
        ask: Ask::Purgeable,
        heaps,
    }
}

/// One heap of each type, for the label chain: a label is a property of the object and not of its
/// memory, so the storage mode is left at one.
fn labels() -> HeapFactsCase {
    HeapFactsCase {
        name: LABELS,
        ask: Ask::Labels,
        heaps: vec![
            of_type(HeapType::Automatic, SIZE, Storage::Shared),
            of_type(HeapType::Placement, SIZE, Storage::Shared),
        ],
    }
}

/// The two buffers the available-size sweep places, in order: an offset and a length each.
///
/// The offsets are multiples of Metal's 256-byte placement grain and both placements lie inside
/// `SIZE`, which [`tests::every_placement_of_the_available_size_sweep_is_one_metal_takes`] holds
/// them to.
const PLACEMENTS: [(u64, u64); 2] = [(0, 4096), (1 << 13, 1 << 16)];

/// The buffer the available-size sweep places in a heap of `heap_type`, by the call that heap's own
/// type takes: the offsetless form of an automatic heap, the offset form of a placement one.
///
/// One function, read by the sweep **and** by the guard that holds it safe. A second copy inside
/// the test would be a mirror rather than a check: it was written that way first, and mutating the
/// sweep's own choice left the test green, which is what a mirror does.
fn placement(heap_type: HeapType, offset: u64, length: u64) -> HeapBufferSpec {
    match heap_type {
        HeapType::Placement => HeapBufferSpec::at(offset, length),
        _ => HeapBufferSpec::suballocated(length),
    }
}

/// A heap of `heap_type`, `size` bytes, in `storage`, Metal's defaults for the other two modes.
fn of_type(heap_type: HeapType, size: u64, storage: Storage) -> HeapSpec {
    match heap_type {
        HeapType::Placement => HeapSpec::placement(size, storage),
        // Every other type reaches `HeapSpec::is_well_formed`, which refuses a sparse one: this
        // family's tables hold none, and `tests` is what says so.
        _ => HeapSpec::automatic(size, storage),
    }
}

/// The sites a case emits.
mod site {
    /// A heap's ten getters, fresh.
    pub const FRESH: &str = "fresh";
    /// The size the case's descriptor asked for, beside what the heap answered.
    pub const ASKED: &str = "size-asked-for";
    /// `maxAvailableSizeWithAlignment:` at one alignment.
    pub const AVAILABLE: &str = "available";
    /// A heap's ten getters after a buffer was placed in it.
    pub const AFTER: &str = "after";
    /// A placed buffer's own `MTLResource` answers.
    pub const PLACED: &str = "placed";
    /// One answer of the purgeable chain.
    pub const PURGEABLE: &str = "purgeable";
    /// A heap's ten getters after one label of the chain was set.
    pub const LABELLED: &str = "labelled";
}

impl Case for HeapFactsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<crate::recording::ShaderRef>, String> {
        // Nothing here compiles: every answer is a getter of a heap or of a buffer placed in one.
        Ok(Vec::new())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The heaps and what the case asks of them, in order: a table that gains, loses or
        // re-points a row is a change to the case's inputs, so a recording cannot decay into the
        // answers of heaps nobody asked about.
        let mut words = Vec::new();
        for heap in &self.heaps {
            words.extend(heap.words());
        }
        let asks: Vec<u8> = match self.ask {
            Ask::OwnGetters => b"getters".to_vec(),
            Ask::AvailableSizes => {
                let mut out = b"maxAvailableSizeWithAlignment:".to_vec();
                for alignment in ALIGNMENTS {
                    out.extend(alignment.to_le_bytes());
                }
                out
            }
            Ask::PlacedBuffers => {
                let mut out = b"newBufferWithLength:options:".to_vec();
                for length in PLACED_LENGTHS {
                    out.extend(HeapBufferSpec::suballocated(length).words());
                }
                out
            }
            Ask::Purgeable => {
                let mut out = b"setPurgeableState:".to_vec();
                for state in PURGEABLE_CHAIN {
                    out.extend(state.raw().to_le_bytes());
                }
                out
            }
            Ask::Labels => {
                let mut out = b"setLabel:".to_vec();
                for label in LABEL_CHAIN {
                    out.extend(label.as_bytes());
                    out.push(0);
                }
                out
            }
        };
        Ok(vec![
            Input::bytes("heaps", &words),
            Input::bytes("asks", &asks),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for spec in &self.heaps {
            let mut node = Node::branch(spec.site(), Vec::new());
            // A heap the venue cannot make is recorded at its own site and the sweep goes on: the
            // other heaps' answers stay readable instead of being lost behind an abort.
            let heap = match side.heap(spec) {
                Ok(heap) => heap,
                Err(failure) => {
                    sites.push(
                        node.with(Node::leaf(site::FRESH, Observed::value(failure.recorded()))),
                    );
                    continue;
                }
            };
            // The size the descriptor asked for, recorded beside what the heap answered: Metal's
            // rounding is the fact, and a reader of the recording should not have to find this
            // module to know what was rounded.
            node = node.with(Node::leaf(
                site::ASKED,
                Observed::value(Value::Uint(spec.size)),
            ));
            node = node.with(own_facts(side, heap, site::FRESH));
            match self.ask {
                Ask::OwnGetters => {}
                Ask::AvailableSizes => node = self.available(side, spec, heap, node),
                Ask::PlacedBuffers => node = self.placed(side, heap, node),
                // Last of everything this case asks of the heap, which is what
                // `crate::heap_query` says the ordering is for: `Empty` lets Metal discard the
                // heap's bytes.
                Ask::Purgeable => node = purgeable(side, heap, node),
                Ask::Labels => node = labelled(side, heap, node),
            }
            sites.push(node);
        }
        Ok(())
    }
}

impl HeapFactsCase {
    /// `maxAvailableSizeWithAlignment:` at every alignment, read fresh and again after each of two
    /// buffers is placed, with the heap's own figures beside each reading.
    ///
    /// Two placements rather than one, because the claim under test is that the answer **moves**:
    /// one reading cannot tell a suballocator from a constant.
    ///
    /// Each heap is placed into **by the call its own type takes**: the offsetless form of an
    /// automatic heap, and the offset form of a placement one. The offsetless form on a placement
    /// heap is a call the runtime refuses `Invalid` — this family's own device guard caught it
    /// there, where the table first held it, which is what the guard is for.
    fn available(
        &self,
        side: &mut dyn Side,
        heap_spec: &HeapSpec,
        heap: Handle,
        mut node: Node,
    ) -> Node {
        node = node.with(available_sizes_at(side, heap, "fresh"));
        for (round, (offset, length)) in PLACEMENTS.into_iter().enumerate() {
            let spec = placement(heap_spec.heap_type, offset, length);
            let at = format!("{}-{}", round, spec.word());
            // A placement the venue cannot make is recorded at its site and the sweep carries on,
            // so the other readings stay readable instead of being lost behind an abort.
            let placed = match side.heap_buffer(heap, &spec) {
                Ok(_) => Value::ok(Value::Absent),
                Err(failure) => failure.recorded(),
            };
            node = node.with(Node::leaf(
                format!("{}/{at}", site::PLACED),
                Observed::value(placed),
            ));
            node = node.with(own_facts(side, heap, &format!("{}/{at}", site::AFTER)));
            node = node.with(available_sizes_at(side, heap, &at));
        }
        node
    }

    /// `newBufferWithLength:options:` at every length of [`PLACED_LENGTHS`], with the heap's
    /// figures and the placed buffer's `heapOffset` after each.
    fn placed(&self, side: &mut dyn Side, heap: Handle, mut node: Node) -> Node {
        for length in PLACED_LENGTHS {
            let spec = HeapBufferSpec::suballocated(length);
            let at = format!("{}/{}", site::PLACED, spec.word());
            // The placed buffer's own `MTLResource` answers go under the placement's site. Its
            // `heapOffset` is 0 on this chip whatever the suballocator did, which is itself the
            // measured fact; what moves is the heap's `usedSize`, read after it.
            let placed = match side.heap_buffer(heap, &spec) {
                Ok(buffer) => {
                    match side.resource_facts(QueriedResource::Buffer(buffer), Some(heap)) {
                        Ok(facts) => facts.node(at.clone()),
                        Err(failure) => Node::leaf(at.clone(), Observed::value(failure.recorded())),
                    }
                }
                Err(failure) => Node::leaf(at.clone(), Observed::value(failure.recorded())),
            };
            node = node.with(placed);
            node = node.with(own_facts(
                side,
                heap,
                &format!("{}/{}", site::AFTER, spec.word()),
            ));
        }
        node
    }
}

/// The heap's ten getters at `at`, as a node of its own.
fn own_facts(side: &mut dyn Side, heap: Handle, at: &str) -> Node {
    match side.heap_own_facts(heap) {
        Ok(facts) => facts.node(at.to_owned()),
        Err(failure) => Node::leaf(at.to_owned(), Observed::value(failure.recorded())),
    }
}

/// `maxAvailableSizeWithAlignment:` at every alignment, under one node named `round`.
fn available_sizes_at(side: &mut dyn Side, heap: Handle, round: &str) -> Node {
    let mut node = Node::branch(format!("{}/{round}", site::AVAILABLE), Vec::new());
    for alignment in ALIGNMENTS {
        let answered = match side.heap_max_available_size(heap, alignment) {
            Ok(size) => Value::ok(Value::Uint(size)),
            Err(failure) => failure.recorded(),
        };
        node = node.with(Node::leaf(
            format!("alignment-{alignment}"),
            Observed::value(answered),
        ));
    }
    node
}

/// The purgeable chain's three answers, each at its own site.
fn purgeable(side: &mut dyn Side, heap: Handle, mut node: Node) -> Node {
    for state in PURGEABLE_CHAIN {
        let answered = match side.heap_purgeable_state(heap, state) {
            Ok(was) => Value::ok(Value::Uint(was)),
            Err(failure) => failure.recorded(),
        };
        node = node.with(Node::leaf(
            format!("{}/{}", site::PURGEABLE, purgeable_word(state)),
            Observed::value(answered),
        ));
    }
    node
}

/// The label chain: each label set, and the heap's getters read back after it.
fn labelled(side: &mut dyn Side, heap: Handle, mut node: Node) -> Node {
    for (round, label) in LABEL_CHAIN.iter().enumerate() {
        if let Err(failure) = side.set_heap_label(heap, label) {
            node = node.with(Node::leaf(
                format!("set-label-{round}"),
                Observed::value(failure.recorded()),
            ));
            continue;
        }
        node = node.with(own_facts(
            side,
            heap,
            &format!("{}/{round}-{}", site::LABELLED, label.len()),
        ));
    }
    node
}

/// The guards this family's safety argument and its claim to measure anything rest on.
///
/// Both run with **no device at all**, which is what a guard against ending Metal's process has to
/// do: a check that needed a run could not say anything about a table before record mode handed it
/// to Metal.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap_query::{cache_word, hazard_word, heap_type_word};
    use metal2vulkan_runtime::{CpuCacheMode, HazardTrackingMode};
    use std::collections::BTreeSet;

    /// No heap of the family is one [`HeapSpec::is_well_formed`] names.
    ///
    /// Verified to fail: making [`of_type`]'s placement arm `HeapSpec::placement(size,
    /// Storage::Managed)` fails here, naming every placement heap of every table and the reason.
    #[test]
    fn no_heap_of_the_family_is_one_metals_validation_stops() {
        let mut stopped = Vec::new();
        for case in all() {
            for heap in &case.heaps {
                if let Some(why) = heap.is_well_formed() {
                    stopped.push(format!("{}: {}: {why}", case.name, heap.site()));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these heaps are ones Metal's validation stops, so record mode would end Metal's \
             process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }

    /// The getter sweep's table moves every field whose answer it records, which is what keeps the
    /// case from passing while measuring nothing.
    ///
    /// A table that only ever stated Metal's defaults would record one `cpuCacheMode` and one
    /// `hazardTrackingMode`, and a side that ignored the descriptor altogether would answer
    /// exactly the same. This is the positive control: it is the reason the recorded words are
    /// words nothing else in the case leaves.
    ///
    /// Verified to fail: taking one mode from each of `CACHE_MODES` and `HAZARD_MODES` in
    /// [`own_getters`]' second loop leaves the table with one of each, and this names it.
    #[test]
    fn the_getter_sweep_moves_every_field_it_records() {
        let case = own_getters();
        let caches: BTreeSet<u64> = case.heaps.iter().map(|heap| heap.cache.raw()).collect();
        let hazards: BTreeSet<u64> = case.heaps.iter().map(|heap| heap.hazard.raw()).collect();
        let types: BTreeSet<u64> = case.heaps.iter().map(|heap| heap.heap_type.raw()).collect();
        let storages: BTreeSet<&'static str> =
            case.heaps.iter().map(|heap| heap.storage.word()).collect();
        assert_eq!(caches.len(), CACHE_MODES.len(), "one CPU cache mode only");
        assert_eq!(
            hazards.len(),
            HAZARD_MODES.len(),
            "not every hazard tracking mode is stated, so what Metal resolves `Default` to cannot \
             be told from what it keeps a stated mode as"
        );
        assert_eq!(types.len(), 2, "one heap type only");
        assert_eq!(storages.len(), BOTH_STORAGE.len(), "one storage mode only");
        assert!(
            case.heaps
                .iter()
                .any(|heap| heap.hazard == HazardTrackingMode::Default),
            "no heap leaves the hazard tracking mode at Metal's default"
        );
        assert!(
            case.heaps
                .iter()
                .any(|heap| heap.cache == CpuCacheMode::WriteCombined),
            "no heap asks for write-combined caching"
        );
    }

    /// No two heaps of a case share a site, which is what keeps one heap's answers from being
    /// compared against another's, and each case fingerprints its own table.
    #[test]
    fn every_heap_names_its_own_site_and_every_case_its_own_inputs() {
        for case in all() {
            let sites: BTreeSet<String> = case.heaps.iter().map(HeapSpec::site).collect();
            assert_eq!(
                sites.len(),
                case.heaps.len(),
                "{}: two heaps share a site",
                case.name
            );
        }
        let fingerprints: BTreeSet<Vec<String>> = all()
            .iter()
            .map(|case| {
                case.inputs()
                    .expect("the inputs are the case's own words")
                    .into_iter()
                    .map(|input| input.digest.hex)
                    .collect()
            })
            .collect();
        assert_eq!(
            fingerprints.len(),
            all().len(),
            "two cases fingerprint alike"
        );
    }

    /// Every placement the available-size sweep makes is one Metal takes: a multiple of the
    /// 256-byte grain, inside the heap, and made by the call its heap's own type takes.
    ///
    /// Verified to fail, three ways: making [`placement`]'s placement arm the offsetless form —
    /// which is how the sweep first stood, and what the device guard caught — fails here naming
    /// the heap type; an offset of 128 in `PLACEMENTS` fails naming the offset; and a length of
    /// `SIZE` at a non-zero offset fails naming the bytes past the heap's end.
    ///
    /// All three reach the guard because the sweep and the guard call **one** function. An earlier
    /// version of this test built the choice again inside itself, and mutating the sweep's own
    /// left it green: that is a mirror, not a check, and it is why [`placement`] exists.
    #[test]
    fn every_placement_of_the_available_size_sweep_is_one_metal_takes() {
        let mut bad = Vec::new();
        for spec in &available_sizes().heaps {
            for (round, (offset, length)) in PLACEMENTS.into_iter().enumerate() {
                // The sweep's own choice, read through the one function it calls -- not a second
                // copy of it, which is what let the first version of this guard stay green while
                // the sweep handed a placement heap the offsetless form.
                let buffer = placement(spec.heap_type, offset, length);
                let is_placement = matches!(spec.heap_type, HeapType::Placement);
                if is_placement != buffer.offset.is_some() {
                    bad.push(format!(
                        "{} round {round}: a {} heap is placed into by {}",
                        spec.site(),
                        heap_type_word(spec.heap_type),
                        buffer.call()
                    ));
                }
                if let Some(offset) = buffer.offset {
                    if !offset.is_multiple_of(256) || offset + buffer.length > SIZE {
                        bad.push(format!(
                            "{} round {round}: {} bytes at {offset}",
                            spec.site(),
                            buffer.length
                        ));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Every alignment the available-size sweep asks is a power of two, which is the bound
    /// [`crate::heap_query`] states the sweep is kept inside.
    ///
    /// Verified to fail: a `3` added to [`ALIGNMENTS`] fails here naming it.
    #[test]
    fn every_alignment_asked_is_a_power_of_two() {
        let bad: Vec<u64> = ALIGNMENTS
            .into_iter()
            .filter(|alignment| !alignment.is_power_of_two())
            .collect();
        assert!(
            bad.is_empty(),
            "these alignments are not powers of two, which the runtime refuses `Invalid` and this \
             tree reads as Metal's validation stopping the call: {bad:?}"
        );
    }

    /// The purgeable chain reads a state back after it set it, which is the only thing that makes
    /// its three answers a transition rule rather than three echoes.
    ///
    /// Verified to fail: a chain of `[Volatile, Volatile, Volatile]` fails here.
    #[test]
    fn the_purgeable_chain_asks_three_different_states_and_ends_in_a_query() {
        let states: BTreeSet<u64> = PURGEABLE_CHAIN.iter().map(|state| state.raw()).collect();
        assert_eq!(
            states.len(),
            PURGEABLE_CHAIN.len(),
            "two calls of the chain hand the same state, so one of them reads nothing new"
        );
        assert_eq!(
            PURGEABLE_CHAIN.last().map(|state| purgeable_word(*state)),
            Some("keep-current".to_owned()),
            "the chain does not end in the query that changes nothing, so its last answer is not \
             a reading of what the call before it did"
        );
    }

    /// The label chain states the empty string and a non-empty one, which is the whole of what it
    /// is for: whether Metal tells a cleared label from one never set.
    #[test]
    fn the_label_chain_states_an_empty_label_and_a_non_empty_one() {
        assert!(LABEL_CHAIN.contains(&""), "no empty label");
        assert!(
            LABEL_CHAIN.iter().any(|label| !label.is_empty()),
            "no non-empty label, so nothing says a set takes at all"
        );
        // The site names carry the label's length, so two rounds of the same label are two sites.
        let sites: BTreeSet<String> = LABEL_CHAIN
            .iter()
            .enumerate()
            .map(|(round, label)| format!("{round}-{}", label.len()))
            .collect();
        assert_eq!(sites.len(), LABEL_CHAIN.len());
    }

    /// The word functions name every mode the family states, and name two modes apart.
    #[test]
    fn the_words_tell_the_modes_apart() {
        let caches: BTreeSet<String> = CACHE_MODES.into_iter().map(cache_word).collect();
        assert_eq!(caches.len(), CACHE_MODES.len());
        let hazards: BTreeSet<String> = HAZARD_MODES.into_iter().map(hazard_word).collect();
        assert_eq!(hazards.len(), HAZARD_MODES.len());
        assert_ne!(
            heap_type_word(HeapType::Automatic),
            heap_type_word(HeapType::Placement)
        );
    }
}

/// The second half of this family's safety argument, run on a device: the **order of work** the
/// heap-size and sparse-heap families state, which is that a table is replayed on the runtime
/// first and only one that answers no `Error::Invalid` anywhere is handed to Metal.
///
/// [`crate::device::sites_the_runtime_calls_invalid`] is the shared walk; this is the family that
/// asks it. The device-free guards above check each row's shape against the rules; this checks the
/// runtime's own reading of every call the rows actually make, including the ones no rule above
/// names.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::case::Case;

    #[test]
    fn no_call_of_the_family_is_one_metals_validation_stops() {
        let cases: Vec<Box<dyn Case>> = all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>)
            .collect();
        let stopped =
            crate::device::sites_the_runtime_calls_invalid(&cases).expect("a device for the guard");
        assert!(
            stopped.is_empty(),
            "the runtime answers `Error::Invalid` at these sites, which is this tree's reading of \
             \"Metal's own validation stops this call\" -- so record mode would end Metal's \
             process. Take the rows behind them out of the tables:\n{}",
            stopped.join("\n")
        );
    }
}
