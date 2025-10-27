static void sdhci_send_command(SDHCIState *s)

{

    SDRequest request;

    uint8_t response[16];

    int rlen;



    s->errintsts = 0;

    s->acmd12errsts = 0;

    request.cmd = s->cmdreg >> 8;

    request.arg = s->argument;

    DPRINT_L1("sending CMD%u ARG[0x%08x]\n", request.cmd, request.arg);

    rlen = sdbus_do_command(&s->sdbus, &request, response);



    if (s->cmdreg & SDHC_CMD_RESPONSE) {

        if (rlen == 4) {

            s->rspreg[0] = (response[0] << 24) | (response[1] << 16) |

                           (response[2] << 8)  |  response[3];

            s->rspreg[1] = s->rspreg[2] = s->rspreg[3] = 0;

            DPRINT_L1("Response: RSPREG[31..0]=0x%08x\n", s->rspreg[0]);

        } else if (rlen == 16) {

            s->rspreg[0] = (response[11] << 24) | (response[12] << 16) |

                           (response[13] << 8) |  response[14];

            s->rspreg[1] = (response[7] << 24) | (response[8] << 16) |

                           (response[9] << 8)  |  response[10];

            s->rspreg[2] = (response[3] << 24) | (response[4] << 16) |

                           (response[5] << 8)  |  response[6];

            s->rspreg[3] = (response[0] << 16) | (response[1] << 8) |

                            response[2];

            DPRINT_L1("Response received:\n RSPREG[127..96]=0x%08x, RSPREG[95.."

                  "64]=0x%08x,\n RSPREG[63..32]=0x%08x, RSPREG[31..0]=0x%08x\n",

                  s->rspreg[3], s->rspreg[2], s->rspreg[1], s->rspreg[0]);

        } else {

            ERRPRINT("Timeout waiting for command response\n");

            if (s->errintstsen & SDHC_EISEN_CMDTIMEOUT) {

                s->errintsts |= SDHC_EIS_CMDTIMEOUT;

                s->norintsts |= SDHC_NIS_ERR;

            }

        }



        if ((s->norintstsen & SDHC_NISEN_TRSCMP) &&

            (s->cmdreg & SDHC_CMD_RESPONSE) == SDHC_CMD_RSP_WITH_BUSY) {

            s->norintsts |= SDHC_NIS_TRSCMP;

        }

    }



    if (s->norintstsen & SDHC_NISEN_CMDCMP) {

        s->norintsts |= SDHC_NIS_CMDCMP;

    }



    sdhci_update_irq(s);



    if (s->blksize && (s->cmdreg & SDHC_CMD_DATA_PRESENT)) {

        s->data_count = 0;

        sdhci_data_transfer(s);

    }

}
