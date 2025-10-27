static void core_commit(MemoryListener *listener)

{

    PhysPageMap info = cur_map;

    cur_map = next_map;

    phys_sections_clear(&info);

}
