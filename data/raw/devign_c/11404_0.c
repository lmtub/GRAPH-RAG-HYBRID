static int cris_mmu_enabled(uint32_t rw_gc_cfg)

{

	return (rw_gc_cfg & 12) != 0;

}
