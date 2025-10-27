static int kvm_log_stop(CPUPhysMemoryClient *client,

                        target_phys_addr_t phys_addr, ram_addr_t size)

{

    return kvm_dirty_pages_log_change(phys_addr, size, false);

}
