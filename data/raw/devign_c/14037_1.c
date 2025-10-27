int target_munmap(target_ulong start, target_ulong len)

{

    target_ulong end, real_start, real_end, addr;

    int prot, ret;



#ifdef DEBUG_MMAP

    printf("munmap: start=0x%lx len=0x%lx\n", start, len);

#endif

    if (start & ~TARGET_PAGE_MASK)

        return -EINVAL;

    len = TARGET_PAGE_ALIGN(len);

    if (len == 0)

        return -EINVAL;

    end = start + len;

    real_start = start & qemu_host_page_mask;

    real_end = HOST_PAGE_ALIGN(end);



    if (start > real_start) {

        /* handle host page containing start */

        prot = 0;

        for(addr = real_start; addr < start; addr += TARGET_PAGE_SIZE) {

            prot |= page_get_flags(addr);

        }

        if (real_end == real_start + qemu_host_page_size) {

            for(addr = end; addr < real_end; addr += TARGET_PAGE_SIZE) {

                prot |= page_get_flags(addr);

            }

            end = real_end;

        }

        if (prot != 0)

            real_start += qemu_host_page_size;

    }

    if (end < real_end) {

        prot = 0;

        for(addr = end; addr < real_end; addr += TARGET_PAGE_SIZE) {

            prot |= page_get_flags(addr);

        }

        if (prot != 0)

            real_end -= qemu_host_page_size;

    }



    /* unmap what we can */

    if (real_start < real_end) {

        ret = munmap((void *)real_start, real_end - real_start);

        if (ret != 0)

            return ret;

    }



    page_set_flags(start, start + len, 0);

    return 0;

}
