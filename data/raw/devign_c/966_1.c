static void megasas_reset_frames(MegasasState *s)

{

    PCIDevice *pcid = PCI_DEVICE(s);

    int i;

    MegasasCmd *cmd;



    for (i = 0; i < s->fw_cmds; i++) {

        cmd = &s->frames[i];

        if (cmd->pa) {

            pci_dma_unmap(pcid, cmd->frame, cmd->pa_size, 0, 0);

            cmd->frame = NULL;

            cmd->pa = 0;

        }

    }

}
