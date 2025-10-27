static int vhost_verify_ring_part_mapping(void *part,

                                          uint64_t part_addr,

                                          uint64_t part_size,

                                          uint64_t start_addr,

                                          uint64_t size)

{

    hwaddr l;

    void *p;

    int r = 0;



    if (!ranges_overlap(start_addr, size, part_addr, part_size)) {

        return 0;

    }

    l = part_size;

    p = cpu_physical_memory_map(part_addr, &l, 1);

    if (!p || l != part_size) {

        r = -ENOMEM;

    }

    if (p != part) {

        r = -EBUSY;

    }

    cpu_physical_memory_unmap(p, l, 0, 0);

    return r;

}
