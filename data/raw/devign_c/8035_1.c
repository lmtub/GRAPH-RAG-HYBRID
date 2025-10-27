void memory_region_init_alias(MemoryRegion *mr,

                              Object *owner,

                              const char *name,

                              MemoryRegion *orig,

                              hwaddr offset,

                              uint64_t size)

{

    memory_region_init(mr, owner, name, size);

    memory_region_ref(orig);

    mr->destructor = memory_region_destructor_alias;

    mr->alias = orig;

    mr->alias_offset = offset;

}
