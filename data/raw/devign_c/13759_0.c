static int ram_load(QEMUFile *f, void *opaque, int version_id)

{

    ram_addr_t addr;

    int flags;



    if (version_id != 3)

        return -EINVAL;



    do {

        addr = qemu_get_be64(f);



        flags = addr & ~TARGET_PAGE_MASK;

        addr &= TARGET_PAGE_MASK;



        if (flags & RAM_SAVE_FLAG_MEM_SIZE) {

            if (addr != last_ram_offset)

                return -EINVAL;

        }



        if (flags & RAM_SAVE_FLAG_COMPRESS) {

            uint8_t ch = qemu_get_byte(f);

            memset(qemu_get_ram_ptr(addr), ch, TARGET_PAGE_SIZE);

#ifndef _WIN32

            if (ch == 0 &&

                (!kvm_enabled() || kvm_has_sync_mmu())) {

                madvise(qemu_get_ram_ptr(addr), TARGET_PAGE_SIZE, MADV_DONTNEED);

            }

#endif

        } else if (flags & RAM_SAVE_FLAG_PAGE) {

            qemu_get_buffer(f, qemu_get_ram_ptr(addr), TARGET_PAGE_SIZE);

        }

        if (qemu_file_has_error(f)) {

            return -EIO;

        }

    } while (!(flags & RAM_SAVE_FLAG_EOS));



    return 0;

}
