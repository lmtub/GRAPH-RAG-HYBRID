void ahci_hba_enable(AHCIQState *ahci)

{

    /* Bits of interest in this section:

     * GHC.AE     Global Host Control / AHCI Enable

     * PxCMD.ST   Port Command: Start

     * PxCMD.SUD  "Spin Up Device"

     * PxCMD.POD  "Power On Device"

     * PxCMD.FRE  "FIS Receive Enable"

     * PxCMD.FR   "FIS Receive Running"

     * PxCMD.CR   "Command List Running"

     */

    uint32_t reg, ports_impl;

    uint16_t i;

    uint8_t num_cmd_slots;



    g_assert(ahci != NULL);



    /* Set GHC.AE to 1 */

    ahci_set(ahci, AHCI_GHC, AHCI_GHC_AE);

    reg = ahci_rreg(ahci, AHCI_GHC);

    ASSERT_BIT_SET(reg, AHCI_GHC_AE);



    /* Cache CAP and CAP2. */

    ahci->cap = ahci_rreg(ahci, AHCI_CAP);

    ahci->cap2 = ahci_rreg(ahci, AHCI_CAP2);



    /* Read CAP.NCS, how many command slots do we have? */

    num_cmd_slots = ((ahci->cap & AHCI_CAP_NCS) >> ctzl(AHCI_CAP_NCS)) + 1;

    g_test_message("Number of Command Slots: %u", num_cmd_slots);



    /* Determine which ports are implemented. */

    ports_impl = ahci_rreg(ahci, AHCI_PI);



    for (i = 0; ports_impl; ports_impl >>= 1, ++i) {

        if (!(ports_impl & 0x01)) {

            continue;

        }



        g_test_message("Initializing port %u", i);



        reg = ahci_px_rreg(ahci, i, AHCI_PX_CMD);

        if (BITCLR(reg, AHCI_PX_CMD_ST | AHCI_PX_CMD_CR |

                   AHCI_PX_CMD_FRE | AHCI_PX_CMD_FR)) {

            g_test_message("port is idle");

        } else {

            g_test_message("port needs to be idled");

            ahci_px_clr(ahci, i, AHCI_PX_CMD,

                        (AHCI_PX_CMD_ST | AHCI_PX_CMD_FRE));

            /* The port has 500ms to disengage. */

            usleep(500000);

            reg = ahci_px_rreg(ahci, i, AHCI_PX_CMD);

            ASSERT_BIT_CLEAR(reg, AHCI_PX_CMD_CR);

            ASSERT_BIT_CLEAR(reg, AHCI_PX_CMD_FR);

            g_test_message("port is now idle");

            /* The spec does allow for possibly needing a PORT RESET

             * or HBA reset if we fail to idle the port. */

        }



        /* Allocate Memory for the Command List Buffer & FIS Buffer */

        /* PxCLB space ... 0x20 per command, as in 4.2.2 p 36 */

        ahci->port[i].clb = ahci_alloc(ahci, num_cmd_slots * 0x20);

        qmemset(ahci->port[i].clb, 0x00, num_cmd_slots * 0x20);

        g_test_message("CLB: 0x%08" PRIx64, ahci->port[i].clb);

        ahci_px_wreg(ahci, i, AHCI_PX_CLB, ahci->port[i].clb);

        g_assert_cmphex(ahci->port[i].clb, ==,

                        ahci_px_rreg(ahci, i, AHCI_PX_CLB));



        /* PxFB space ... 0x100, as in 4.2.1 p 35 */

        ahci->port[i].fb = ahci_alloc(ahci, 0x100);

        qmemset(ahci->port[i].fb, 0x00, 0x100);

        g_test_message("FB: 0x%08" PRIx64, ahci->port[i].fb);

        ahci_px_wreg(ahci, i, AHCI_PX_FB, ahci->port[i].fb);

        g_assert_cmphex(ahci->port[i].fb, ==,

                        ahci_px_rreg(ahci, i, AHCI_PX_FB));



        /* Clear PxSERR, PxIS, then IS.IPS[x] by writing '1's. */

        ahci_px_wreg(ahci, i, AHCI_PX_SERR, 0xFFFFFFFF);

        ahci_px_wreg(ahci, i, AHCI_PX_IS, 0xFFFFFFFF);

        ahci_wreg(ahci, AHCI_IS, (1 << i));



        /* Verify Interrupts Cleared */

        reg = ahci_px_rreg(ahci, i, AHCI_PX_SERR);

        g_assert_cmphex(reg, ==, 0);



        reg = ahci_px_rreg(ahci, i, AHCI_PX_IS);

        g_assert_cmphex(reg, ==, 0);



        reg = ahci_rreg(ahci, AHCI_IS);

        ASSERT_BIT_CLEAR(reg, (1 << i));



        /* Enable All Interrupts: */

        ahci_px_wreg(ahci, i, AHCI_PX_IE, 0xFFFFFFFF);

        reg = ahci_px_rreg(ahci, i, AHCI_PX_IE);

        g_assert_cmphex(reg, ==, ~((uint32_t)AHCI_PX_IE_RESERVED));



        /* Enable the FIS Receive Engine. */

        ahci_px_set(ahci, i, AHCI_PX_CMD, AHCI_PX_CMD_FRE);

        reg = ahci_px_rreg(ahci, i, AHCI_PX_CMD);

        ASSERT_BIT_SET(reg, AHCI_PX_CMD_FR);



        /* AHCI 1.3 spec: if !STS.BSY, !STS.DRQ and PxSSTS.DET indicates

         * physical presence, a device is present and may be started. However,

         * PxSERR.DIAG.X /may/ need to be cleared a priori. */

        reg = ahci_px_rreg(ahci, i, AHCI_PX_SERR);

        if (BITSET(reg, AHCI_PX_SERR_DIAG_X)) {

            ahci_px_set(ahci, i, AHCI_PX_SERR, AHCI_PX_SERR_DIAG_X);

        }



        reg = ahci_px_rreg(ahci, i, AHCI_PX_TFD);

        if (BITCLR(reg, AHCI_PX_TFD_STS_BSY | AHCI_PX_TFD_STS_DRQ)) {

            reg = ahci_px_rreg(ahci, i, AHCI_PX_SSTS);

            if ((reg & AHCI_PX_SSTS_DET) == SSTS_DET_ESTABLISHED) {

                /* Device Found: set PxCMD.ST := 1 */

                ahci_px_set(ahci, i, AHCI_PX_CMD, AHCI_PX_CMD_ST);

                ASSERT_BIT_SET(ahci_px_rreg(ahci, i, AHCI_PX_CMD),

                               AHCI_PX_CMD_CR);

                g_test_message("Started Device %u", i);

            } else if ((reg & AHCI_PX_SSTS_DET)) {

                /* Device present, but in some unknown state. */

                g_assert_not_reached();

            }

        }

    }



    /* Enable GHC.IE */

    ahci_set(ahci, AHCI_GHC, AHCI_GHC_IE);

    reg = ahci_rreg(ahci, AHCI_GHC);

    ASSERT_BIT_SET(reg, AHCI_GHC_IE);




    /* TODO: The device should now be idling and waiting for commands.

     * In the future, a small test-case to inspect the Register D2H FIS

     * and clear the initial interrupts might be good. */

}