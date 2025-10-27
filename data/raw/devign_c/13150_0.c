static inline void kqemu_load_seg(struct kqemu_segment_cache *ksc,

                                  const SegmentCache *sc)

{

    ksc->selector = sc->selector;

    ksc->flags = sc->flags;

    ksc->limit = sc->limit;

    ksc->base = sc->base;

}
