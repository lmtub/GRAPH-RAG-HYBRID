void address_space_unmap(AddressSpace *as, void *buffer, hwaddr len,

                         int is_write, hwaddr access_len)

{

    if (buffer != bounce.buffer) {

        if (is_write) {

            ram_addr_t addr1 = qemu_ram_addr_from_host_nofail(buffer);

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

        return;

    }

    if (is_write) {

        address_space_write(as, bounce.addr, bounce.buffer, access_len);

    }

    qemu_vfree(bounce.buffer);

    bounce.buffer = NULL;

    cpu_notify_map_clients();

}
