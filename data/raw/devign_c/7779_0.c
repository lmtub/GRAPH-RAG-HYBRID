static void megasas_unmap_frame(MegasasState *s, MegasasCmd *cmd)

{

    PCIDevice *p = PCI_DEVICE(s);



    pci_dma_unmap(p, cmd->frame, cmd->pa_size, 0, 0);

    cmd->frame = NULL;

    cmd->pa = 0;

    clear_bit(cmd->index, s->frame_map);

}
