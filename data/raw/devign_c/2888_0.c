static inline void load_seg_cache_raw_dt(SegmentCache *sc, uint32_t e1, uint32_t e2)

{

    sc->base = get_seg_base(e1, e2);

    sc->limit = get_seg_limit(e1, e2);

    sc->flags = e2;

}
