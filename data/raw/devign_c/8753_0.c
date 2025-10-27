static int kvm_client_sync_dirty_bitmap(struct CPUPhysMemoryClient *client,

					target_phys_addr_t start_addr,

					target_phys_addr_t end_addr)

{

	return kvm_physical_sync_dirty_bitmap(start_addr, end_addr);

}
