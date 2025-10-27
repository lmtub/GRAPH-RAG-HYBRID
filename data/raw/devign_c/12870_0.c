void kvm_set_phys_mem(target_phys_addr_t start_addr,

                      ram_addr_t size,

                      ram_addr_t phys_offset)

{

    KVMState *s = kvm_state;

    ram_addr_t flags = phys_offset & ~TARGET_PAGE_MASK;

    KVMSlot *mem;



    /* KVM does not support read-only slots */

    phys_offset &= ~IO_MEM_ROM;



    mem = kvm_lookup_slot(s, start_addr);

    if (mem) {

        if (flags == IO_MEM_UNASSIGNED) {

            mem->memory_size = 0;

            mem->guest_phys_addr = start_addr;

            mem->userspace_addr = 0;

            mem->flags = 0;



            kvm_vm_ioctl(s, KVM_SET_USER_MEMORY_REGION, mem);

        } else if (start_addr >= mem->guest_phys_addr &&

                   (start_addr + size) <= (mem->guest_phys_addr + mem->memory_size))

            return;

    }



    /* KVM does not need to know about this memory */

    if (flags >= IO_MEM_UNASSIGNED)

        return;



    mem = kvm_alloc_slot(s);

    mem->memory_size = size;

    mem->guest_phys_addr = start_addr;

    mem->userspace_addr = (unsigned long)(phys_ram_base + phys_offset);

    mem->flags = 0;



    kvm_vm_ioctl(s, KVM_SET_USER_MEMORY_REGION, mem);

    /* FIXME deal with errors */

}
