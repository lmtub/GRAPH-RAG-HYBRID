static void eepro100_cu_command(EEPRO100State * s, uint8_t val)

{

    eepro100_tx_t tx;

    uint32_t cb_address;

    switch (val) {

    case CU_NOP:

        /* No operation. */

        break;

    case CU_START:

        if (get_cu_state(s) != cu_idle) {

            /* Intel documentation says that CU must be idle for the CU

             * start command. Intel driver for Linux also starts the CU

             * from suspended state. */

            logout("CU state is %u, should be %u\n", get_cu_state(s), cu_idle);

            //~ assert(!"wrong CU state");

        }

        set_cu_state(s, cu_active);

        s->cu_offset = s->pointer;

      next_command:

        cb_address = s->cu_base + s->cu_offset;

        cpu_physical_memory_read(cb_address, (uint8_t *) & tx, sizeof(tx));

        uint16_t status = le16_to_cpu(tx.status);

        uint16_t command = le16_to_cpu(tx.command);

        logout

            ("val=0x%02x (cu start), status=0x%04x, command=0x%04x, link=0x%08x\n",

             val, status, command, tx.link);

        bool bit_el = ((command & 0x8000) != 0);

        bool bit_s = ((command & 0x4000) != 0);

        bool bit_i = ((command & 0x2000) != 0);

        bool bit_nc = ((command & 0x0010) != 0);

        bool success = true;

        //~ bool bit_sf = ((command & 0x0008) != 0);

        uint16_t cmd = command & 0x0007;

        s->cu_offset = le32_to_cpu(tx.link);

        switch (cmd) {

        case CmdNOp:

            /* Do nothing. */

            break;

        case CmdIASetup:

            cpu_physical_memory_read(cb_address + 8, &s->macaddr[0], 6);

            TRACE(OTHER, logout("macaddr: %s\n", nic_dump(&s->macaddr[0], 6)));

            break;

        case CmdConfigure:

            cpu_physical_memory_read(cb_address + 8, &s->configuration[0],

                                     sizeof(s->configuration));

            TRACE(OTHER, logout("configuration: %s\n", nic_dump(&s->configuration[0], 16)));

            break;

        case CmdMulticastList:

            //~ missing("multicast list");

            break;

        case CmdTx:

            (void)0;

            uint32_t tbd_array = le32_to_cpu(tx.tx_desc_addr);

            uint16_t tcb_bytes = (le16_to_cpu(tx.tcb_bytes) & 0x3fff);

            TRACE(RXTX, logout

                ("transmit, TBD array address 0x%08x, TCB byte count 0x%04x, TBD count %u\n",

                 tbd_array, tcb_bytes, tx.tbd_count));



            if (bit_nc) {

                missing("CmdTx: NC = 0");

                success = false;

                break;

            }

            //~ assert(!bit_sf);

            if (tcb_bytes > 2600) {

                logout("TCB byte count too large, using 2600\n");

                tcb_bytes = 2600;

            }

            /* Next assertion fails for local configuration. */

            //~ assert((tcb_bytes > 0) || (tbd_array != 0xffffffff));

            if (!((tcb_bytes > 0) || (tbd_array != 0xffffffff))) {

                logout

                    ("illegal values of TBD array address and TCB byte count!\n");

            }

            // sends larger than MAX_ETH_FRAME_SIZE are allowed, up to 2600 bytes

            uint8_t buf[2600];

            uint16_t size = 0;

            uint32_t tbd_address = cb_address + 0x10;

            assert(tcb_bytes <= sizeof(buf));

            while (size < tcb_bytes) {

                uint32_t tx_buffer_address = ldl_phys(tbd_address);

                uint16_t tx_buffer_size = lduw_phys(tbd_address + 4);

                //~ uint16_t tx_buffer_el = lduw_phys(tbd_address + 6);

                tbd_address += 8;

                TRACE(RXTX, logout

                    ("TBD (simplified mode): buffer address 0x%08x, size 0x%04x\n",

                     tx_buffer_address, tx_buffer_size));

                tx_buffer_size = MIN(tx_buffer_size, sizeof(buf) - size);

                cpu_physical_memory_read(tx_buffer_address, &buf[size],

                                         tx_buffer_size);

                size += tx_buffer_size;

            }

            if (tbd_array == 0xffffffff) {

                /* Simplified mode. Was already handled by code above. */

            } else {

                /* Flexible mode. */

                uint8_t tbd_count = 0;

                if (device_supports_eTxCB(s) && !(s->configuration[6] & BIT(4))) {

                    /* Extended Flexible TCB. */

                    for (; tbd_count < 2; tbd_count++) {

                        uint32_t tx_buffer_address = ldl_phys(tbd_address);

                        uint16_t tx_buffer_size = lduw_phys(tbd_address + 4);

                        uint16_t tx_buffer_el = lduw_phys(tbd_address + 6);

                        tbd_address += 8;

                        TRACE(RXTX, logout

                            ("TBD (extended flexible mode): buffer address 0x%08x, size 0x%04x\n",

                             tx_buffer_address, tx_buffer_size));

                        tx_buffer_size = MIN(tx_buffer_size, sizeof(buf) - size);

                        cpu_physical_memory_read(tx_buffer_address, &buf[size],

                                                 tx_buffer_size);

                        size += tx_buffer_size;

                        if (tx_buffer_el & 1) {

                            break;

                        }

                    }

                }

                tbd_address = tbd_array;

                for (; tbd_count < tx.tbd_count; tbd_count++) {

                    uint32_t tx_buffer_address = ldl_phys(tbd_address);

                    uint16_t tx_buffer_size = lduw_phys(tbd_address + 4);

                    uint16_t tx_buffer_el = lduw_phys(tbd_address + 6);

                    tbd_address += 8;

                    TRACE(RXTX, logout

                        ("TBD (flexible mode): buffer address 0x%08x, size 0x%04x\n",

                         tx_buffer_address, tx_buffer_size));

                    tx_buffer_size = MIN(tx_buffer_size, sizeof(buf) - size);

                    cpu_physical_memory_read(tx_buffer_address, &buf[size],

                                             tx_buffer_size);

                    size += tx_buffer_size;

                    if (tx_buffer_el & 1) {

                        break;

                    }

                }

            }

            TRACE(RXTX, logout("%p sending frame, len=%d,%s\n", s, size, nic_dump(buf, size)));

            qemu_send_packet(s->vc, buf, size);

            s->statistics.tx_good_frames++;

            /* Transmit with bad status would raise an CX/TNO interrupt.

             * (82557 only). Emulation never has bad status. */

            //~ eepro100_cx_interrupt(s);

            break;

        case CmdTDR:

            TRACE(OTHER, logout("load microcode\n"));

            /* Starting with offset 8, the command contains

             * 64 dwords microcode which we just ignore here. */

            break;

        default:

            missing("undefined command");

            success = false;

            break;

        }

        /* Write new status. */

        stw_phys(cb_address, status | 0x8000 | (success ? 0x2000 : 0));

        if (bit_i) {

            /* CU completed action. */

            eepro100_cx_interrupt(s);

        }

        if (bit_el) {

            /* CU becomes idle. Terminate command loop. */

            set_cu_state(s, cu_idle);

            eepro100_cna_interrupt(s);

        } else if (bit_s) {

            /* CU becomes suspended. */

            set_cu_state(s, cu_suspended);

            eepro100_cna_interrupt(s);

        } else {

            /* More entries in list. */

            TRACE(OTHER, logout("CU list with at least one more entry\n"));

            goto next_command;

        }

        TRACE(OTHER, logout("CU list empty\n"));

        /* List is empty. Now CU is idle or suspended. */

        break;

    case CU_RESUME:

        if (get_cu_state(s) != cu_suspended) {

            logout("bad CU resume from CU state %u\n", get_cu_state(s));

            /* Workaround for bad Linux eepro100 driver which resumes

             * from idle state. */

            //~ missing("cu resume");

            set_cu_state(s, cu_suspended);

        }

        if (get_cu_state(s) == cu_suspended) {

            TRACE(OTHER, logout("CU resuming\n"));

            set_cu_state(s, cu_active);

            goto next_command;

        }

        break;

    case CU_STATSADDR:

        /* Load dump counters address. */

        s->statsaddr = s->pointer;

        TRACE(OTHER, logout("val=0x%02x (status address)\n", val));

        break;

    case CU_SHOWSTATS:

        /* Dump statistical counters. */

        TRACE(OTHER, logout("val=0x%02x (dump stats)\n", val));

        dump_statistics(s);

        break;

    case CU_CMD_BASE:

        /* Load CU base. */

        TRACE(OTHER, logout("val=0x%02x (CU base address)\n", val));

        s->cu_base = s->pointer;

        break;

    case CU_DUMPSTATS:

        /* Dump and reset statistical counters. */

        TRACE(OTHER, logout("val=0x%02x (dump stats and reset)\n", val));

        dump_statistics(s);

        memset(&s->statistics, 0, sizeof(s->statistics));

        break;

    case CU_SRESUME:

        /* CU static resume. */

        missing("CU static resume");

        break;

    default:

        missing("Undefined CU command");

    }

}
