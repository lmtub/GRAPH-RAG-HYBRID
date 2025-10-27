void address_space_unmap(AddressSpace *as, void *buffer, hwaddr len,

                         int is_write, hwaddr access_len)

{

    if (buffer != bounce.buffer) {

        MemoryRegion *mr;

        ram_addr_t addr1;



        mr = qemu_ram_addr_from_host(buffer, &addr1);

        assert(mr != NULL);

        if (is_write) {

            while (access_len) {

                unsigned l;

                l = TARGET_PAGE_SIZE;

                if (l > access_len)

                    l = access_len;

                invalidate_and_set_dirty(addr1, l);

                addr1 += l;

                access_len -= l;

            }

        }

        if (xen_enabled()) {

            xen_invalidate_map_cache_entry(buffer);

        }

        memory_region_unref(mr);

        return;

    }

    if (is_write) {

        address_space_write(as, bounce.addr, bounce.buffer, access_len);

    }

    qemu_vfree(bounce.buffer);

    bounce.buffer = NULL;

    memory_region_unref(bounce.mr);

    cpu_notify_map_clients();

}
