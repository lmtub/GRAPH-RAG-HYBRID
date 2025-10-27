void kvm_physical_sync_dirty_bitmap(target_phys_addr_t start_addr, target_phys_addr_t end_addr)

{

    KVMState *s = kvm_state;

    KVMDirtyLog d;

    KVMSlot *mem = kvm_lookup_slot(s, start_addr);

    unsigned long alloc_size;

    ram_addr_t addr;

    target_phys_addr_t phys_addr = start_addr;



    dprintf("sync addr: %llx into %lx\n", start_addr, mem->phys_offset);

    if (mem == NULL) {

            fprintf(stderr, "BUG: %s: invalid parameters\n", __func__);

            return;

    }



    alloc_size = mem->memory_size >> TARGET_PAGE_BITS / sizeof(d.dirty_bitmap);

    d.dirty_bitmap = qemu_mallocz(alloc_size);



    d.slot = mem->slot;

    dprintf("slot %d, phys_addr %llx, uaddr: %llx\n",

            d.slot, mem->start_addr, mem->phys_offset);



    if (kvm_vm_ioctl(s, KVM_GET_DIRTY_LOG, &d) == -1) {

        dprintf("ioctl failed %d\n", errno);

        goto out;

    }



    phys_addr = start_addr;

    for (addr = mem->phys_offset; phys_addr < end_addr; phys_addr+= TARGET_PAGE_SIZE, addr += TARGET_PAGE_SIZE) {

        unsigned long *bitmap = (unsigned long *)d.dirty_bitmap;

        unsigned nr = (phys_addr - start_addr) >> TARGET_PAGE_BITS;

        unsigned word = nr / (sizeof(*bitmap) * 8);

        unsigned bit = nr % (sizeof(*bitmap) * 8);

        if ((bitmap[word] >> bit) & 1)

            cpu_physical_memory_set_dirty(addr);

    }

out:

    qemu_free(d.dirty_bitmap);

}
