void *etraxfs_eth_init(NICInfo *nd, CPUState *env, 

		       qemu_irq *irq, target_phys_addr_t base, int phyaddr)

{

	struct etraxfs_dma_client *dma = NULL;	

	struct fs_eth *eth = NULL;



	qemu_check_nic_model(nd, "fseth");



	dma = qemu_mallocz(sizeof *dma * 2);



	eth = qemu_mallocz(sizeof *eth);



	dma[0].client.push = eth_tx_push;

	dma[0].client.opaque = eth;

	dma[1].client.opaque = eth;

	dma[1].client.pull = NULL;



	eth->env = env;

	eth->irq = irq;

	eth->dma_out = dma;

	eth->dma_in = dma + 1;



	/* Connect the phy.  */

	eth->phyaddr = phyaddr & 0x1f;

	tdk_init(&eth->phy);

	mdio_attach(&eth->mdio_bus, &eth->phy, eth->phyaddr);



	eth->ethregs = cpu_register_io_memory(0, eth_read, eth_write, eth);

	cpu_register_physical_memory (base, 0x5c, eth->ethregs);



	eth->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

				       eth_receive, eth_can_receive, eth);

	eth->vc->opaque = eth;

	eth->vc->link_status_changed = eth_set_link;



	return dma;

}
