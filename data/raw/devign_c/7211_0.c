static void kvm_client_set_memory(struct CPUPhysMemoryClient *client,

				  target_phys_addr_t start_addr,

				  ram_addr_t size,

				  ram_addr_t phys_offset)

{

	kvm_set_phys_mem(start_addr, size, phys_offset);

}
