static void process_ncq_command(AHCIState *s, int port, uint8_t *cmd_fis,

                                int slot)

{

    AHCIDevice *ad = &s->dev[port];

    IDEState *ide_state = &ad->port.ifs[0];

    NCQFrame *ncq_fis = (NCQFrame*)cmd_fis;

    uint8_t tag = ncq_fis->tag >> 3;

    NCQTransferState *ncq_tfs = &ad->ncq_tfs[tag];

    size_t size;



    if (ncq_tfs->used) {

        /* error - already in use */

        fprintf(stderr, "%s: tag %d already used\n", __FUNCTION__, tag);

        return;

    }



    ncq_tfs->used = 1;

    ncq_tfs->drive = ad;

    ncq_tfs->slot = slot;

    ncq_tfs->cmd = ncq_fis->command;

    ncq_tfs->lba = ((uint64_t)ncq_fis->lba5 << 40) |

                   ((uint64_t)ncq_fis->lba4 << 32) |

                   ((uint64_t)ncq_fis->lba3 << 24) |

                   ((uint64_t)ncq_fis->lba2 << 16) |

                   ((uint64_t)ncq_fis->lba1 << 8) |

                   (uint64_t)ncq_fis->lba0;

    ncq_tfs->tag = tag;



    /* Sanity-check the NCQ packet */

    if (tag != slot) {

        DPRINTF(port, "Warn: NCQ slot (%d) did not match the given tag (%d)\n",

                slot, tag);

    }



    if (ncq_fis->aux0 || ncq_fis->aux1 || ncq_fis->aux2 || ncq_fis->aux3) {

        DPRINTF(port, "Warn: Attempt to use NCQ auxiliary fields.\n");

    }

    if (ncq_fis->prio || ncq_fis->icc) {

        DPRINTF(port, "Warn: Unsupported attempt to use PRIO/ICC fields\n");

    }

    if (ncq_fis->fua & NCQ_FIS_FUA_MASK) {

        DPRINTF(port, "Warn: Unsupported attempt to use Force Unit Access\n");

    }

    if (ncq_fis->tag & NCQ_FIS_RARC_MASK) {

        DPRINTF(port, "Warn: Unsupported attempt to use Rebuild Assist\n");

    }



    ncq_tfs->sector_count = ((uint16_t)ncq_fis->sector_count_high << 8) |

                                ncq_fis->sector_count_low;

    size = ncq_tfs->sector_count * 512;

    ahci_populate_sglist(ad, &ncq_tfs->sglist, size, 0);



    if (ncq_tfs->sglist.size < size) {

        error_report("ahci: PRDT length for NCQ command (0x%zx) "

                     "is smaller than the requested size (0x%zx)",

                     ncq_tfs->sglist.size, size);

        qemu_sglist_destroy(&ncq_tfs->sglist);

        ncq_err(ncq_tfs);

        ahci_trigger_irq(ad->hba, ad, PORT_IRQ_OVERFLOW);

        return;

    } else if (ncq_tfs->sglist.size != size) {

        DPRINTF(port, "Warn: PRDTL (0x%zx)"

                " does not match requested size (0x%zx)",

                ncq_tfs->sglist.size, size);

    }



    DPRINTF(port, "NCQ transfer LBA from %"PRId64" to %"PRId64", "

            "drive max %"PRId64"\n",

            ncq_tfs->lba, ncq_tfs->lba + ncq_tfs->sector_count - 1,

            ide_state->nb_sectors - 1);



    switch (ncq_tfs->cmd) {

        case READ_FPDMA_QUEUED:

            DPRINTF(port, "NCQ reading %d sectors from LBA %"PRId64", "

                    "tag %d\n",

                    ncq_tfs->sector_count, ncq_tfs->lba, ncq_tfs->tag);



            DPRINTF(port, "tag %d aio read %"PRId64"\n",

                    ncq_tfs->tag, ncq_tfs->lba);



            dma_acct_start(ide_state->blk, &ncq_tfs->acct,

                           &ncq_tfs->sglist, BLOCK_ACCT_READ);

            ncq_tfs->aiocb = dma_blk_read(ide_state->blk,

                                          &ncq_tfs->sglist, ncq_tfs->lba,

                                          ncq_cb, ncq_tfs);

            break;

        case WRITE_FPDMA_QUEUED:

            DPRINTF(port, "NCQ writing %d sectors to LBA %"PRId64", tag %d\n",

                    ncq_tfs->sector_count, ncq_tfs->lba, ncq_tfs->tag);



            DPRINTF(port, "tag %d aio write %"PRId64"\n",

                    ncq_tfs->tag, ncq_tfs->lba);



            dma_acct_start(ide_state->blk, &ncq_tfs->acct,

                           &ncq_tfs->sglist, BLOCK_ACCT_WRITE);

            ncq_tfs->aiocb = dma_blk_write(ide_state->blk,

                                           &ncq_tfs->sglist, ncq_tfs->lba,

                                           ncq_cb, ncq_tfs);

            break;

        default:

            if (is_ncq(cmd_fis[2])) {

                DPRINTF(port,

                        "error: unsupported NCQ command (0x%02x) received\n",

                        cmd_fis[2]);

            } else {

                DPRINTF(port,

                        "error: tried to process non-NCQ command as NCQ\n");

            }

            qemu_sglist_destroy(&ncq_tfs->sglist);

            ncq_err(ncq_tfs);

    }

}
