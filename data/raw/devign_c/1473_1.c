static int kvm_physical_sync_dirty_bitmap(target_phys_addr_t start_addr,

                                          target_phys_addr_t end_addr)

{

    KVMState *s = kvm_state;

    unsigned long size, allocated_size = 0;

    KVMDirtyLog d;

    KVMSlot *mem;

    int ret = 0;



    d.dirty_bitmap = NULL;

    while (start_addr < end_addr) {

        mem = kvm_lookup_overlapping_slot(s, start_addr, end_addr);

        if (mem == NULL) {

            break;

        }



        size = ALIGN(((mem->memory_size) >> TARGET_PAGE_BITS), HOST_LONG_BITS) / 8;

        if (!d.dirty_bitmap) {

            d.dirty_bitmap = qemu_malloc(size);

        } else if (size > allocated_size) {

            d.dirty_bitmap = qemu_realloc(d.dirty_bitmap, size);

        }

        allocated_size = size;

        memset(d.dirty_bitmap, 0, allocated_size);



        d.slot = mem->slot;



        if (kvm_vm_ioctl(s, KVM_GET_DIRTY_LOG, &d) == -1) {

            DPRINTF("ioctl failed %d\n", errno);

            ret = -1;

            break;

        }



        kvm_get_dirty_pages_log_range(mem->start_addr, d.dirty_bitmap,

                                      mem->start_addr, mem->memory_size);

        start_addr = mem->start_addr + mem->memory_size;

    }

    qemu_free(d.dirty_bitmap);



    return ret;

}
