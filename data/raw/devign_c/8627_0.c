static int kvm_physical_sync_dirty_bitmap(MemoryRegionSection *section)

{

    KVMState *s = kvm_state;

    unsigned long size, allocated_size = 0;

    KVMDirtyLog d;

    KVMSlot *mem;

    int ret = 0;

    hwaddr start_addr = section->offset_within_address_space;

    hwaddr end_addr = start_addr + int128_get64(section->size);



    d.dirty_bitmap = NULL;

    while (start_addr < end_addr) {

        mem = kvm_lookup_overlapping_slot(s, start_addr, end_addr);

        if (mem == NULL) {

            break;

        }



        /* XXX bad kernel interface alert

         * For dirty bitmap, kernel allocates array of size aligned to

         * bits-per-long.  But for case when the kernel is 64bits and

         * the userspace is 32bits, userspace can't align to the same

         * bits-per-long, since sizeof(long) is different between kernel

         * and user space.  This way, userspace will provide buffer which

         * may be 4 bytes less than the kernel will use, resulting in

         * userspace memory corruption (which is not detectable by valgrind

         * too, in most cases).

         * So for now, let's align to 64 instead of HOST_LONG_BITS here, in

         * a hope that sizeof(long) wont become >8 any time soon.

         */

        size = ALIGN(((mem->memory_size) >> TARGET_PAGE_BITS),

                     /*HOST_LONG_BITS*/ 64) / 8;

        if (!d.dirty_bitmap) {

            d.dirty_bitmap = g_malloc(size);

        } else if (size > allocated_size) {

            d.dirty_bitmap = g_realloc(d.dirty_bitmap, size);

        }

        allocated_size = size;

        memset(d.dirty_bitmap, 0, allocated_size);



        d.slot = mem->slot;



        if (kvm_vm_ioctl(s, KVM_GET_DIRTY_LOG, &d) == -1) {

            DPRINTF("ioctl failed %d\n", errno);

            ret = -1;

            break;

        }



        kvm_get_dirty_pages_log_range(section, d.dirty_bitmap);

        start_addr = mem->start_addr + mem->memory_size;

    }

    g_free(d.dirty_bitmap);



    return ret;

}
