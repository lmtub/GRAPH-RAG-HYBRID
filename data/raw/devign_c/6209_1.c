ds1225y_t *ds1225y_init(target_phys_addr_t mem_base, const char *filename)

{

    ds1225y_t *s;

    int mem_index1, mem_index2;



    s = qemu_mallocz(sizeof(ds1225y_t));

    if (!s)

        return NULL;

    s->mem_base = mem_base;

    s->capacity = 0x2000; /* Fixed for ds1225y chip: 8K */

    s->filename = filename;



    /* Read/write memory */

    mem_index1 = cpu_register_io_memory(0, nvram_read, nvram_write, s);

    cpu_register_physical_memory(mem_base, s->capacity, mem_index1);

    /* Read-only memory */

    mem_index2 = cpu_register_io_memory(0, nvram_read, nvram_none, s);

    cpu_register_physical_memory(mem_base + s->capacity, s->capacity, mem_index2);

    return s;

}
