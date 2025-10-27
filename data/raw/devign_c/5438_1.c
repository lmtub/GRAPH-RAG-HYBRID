void memory_region_destroy(MemoryRegion *mr)

{

    assert(QTAILQ_EMPTY(&mr->subregions));


    mr->destructor(mr);

    memory_region_clear_coalescing(mr);

    g_free((char *)mr->name);

    g_free(mr->ioeventfds);

}