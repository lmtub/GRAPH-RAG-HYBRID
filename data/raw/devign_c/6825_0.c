void ahci_command_wait(AHCIQState *ahci, AHCICommand *cmd)

{

    /* We can't rely on STS_BSY until the command has started processing.

     * Therefore, we also use the Command Issue bit as indication of

     * a command in-flight. */

    while (BITSET(ahci_px_rreg(ahci, cmd->port, AHCI_PX_TFD),

                  AHCI_PX_TFD_STS_BSY) ||

           BITSET(ahci_px_rreg(ahci, cmd->port, AHCI_PX_CI), (1 << cmd->slot))) {

        usleep(50);

    }

}
