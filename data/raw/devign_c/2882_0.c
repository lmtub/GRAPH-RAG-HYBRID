static void tpm_tis_mmio_write_intern(void *opaque, hwaddr addr,

                                      uint64_t val, unsigned size,

                                      bool hw_access)

{

    TPMState *s = opaque;

    TPMTISEmuState *tis = &s->s.tis;

    uint16_t off = addr & 0xffc;

    uint8_t shift = (addr & 0x3) * 8;

    uint8_t locty = tpm_tis_locality_from_addr(addr);

    uint8_t active_locty, l;

    int c, set_new_locty = 1;

    uint16_t len;

    uint32_t mask = (size == 1) ? 0xff : ((size == 2) ? 0xffff : ~0);



    DPRINTF("tpm_tis: write.%u(%08x) = %08x\n", size, (int)addr, (uint32_t)val);



    if (locty == 4 && !hw_access) {

        DPRINTF("tpm_tis: Access to locality 4 only allowed from hardware\n");

        return;

    }



    if (tpm_backend_had_startup_error(s->be_driver)) {

        return;

    }



    val &= mask;



    if (shift) {

        val <<= shift;

        mask <<= shift;

    }



    mask ^= 0xffffffff;



    switch (off) {

    case TPM_TIS_REG_ACCESS:



        if ((val & TPM_TIS_ACCESS_SEIZE)) {

            val &= ~(TPM_TIS_ACCESS_REQUEST_USE |

                     TPM_TIS_ACCESS_ACTIVE_LOCALITY);

        }



        active_locty = tis->active_locty;



        if ((val & TPM_TIS_ACCESS_ACTIVE_LOCALITY)) {

            /* give up locality if currently owned */

            if (tis->active_locty == locty) {

                DPRINTF("tpm_tis: Releasing locality %d\n", locty);



                uint8_t newlocty = TPM_TIS_NO_LOCALITY;

                /* anybody wants the locality ? */

                for (c = TPM_TIS_NUM_LOCALITIES - 1; c >= 0; c--) {

                    if ((tis->loc[c].access & TPM_TIS_ACCESS_REQUEST_USE)) {

                        DPRINTF("tpm_tis: Locality %d requests use.\n", c);

                        newlocty = c;

                        break;

                    }

                }

                DPRINTF("tpm_tis: TPM_TIS_ACCESS_ACTIVE_LOCALITY: "

                        "Next active locality: %d\n",

                        newlocty);



                if (TPM_TIS_IS_VALID_LOCTY(newlocty)) {

                    set_new_locty = 0;

                    tpm_tis_prep_abort(s, locty, newlocty);

                } else {

                    active_locty = TPM_TIS_NO_LOCALITY;

                }

            } else {

                /* not currently the owner; clear a pending request */

                tis->loc[locty].access &= ~TPM_TIS_ACCESS_REQUEST_USE;

            }

        }



        if ((val & TPM_TIS_ACCESS_BEEN_SEIZED)) {

            tis->loc[locty].access &= ~TPM_TIS_ACCESS_BEEN_SEIZED;

        }



        if ((val & TPM_TIS_ACCESS_SEIZE)) {

            /*

             * allow seize if a locality is active and the requesting

             * locality is higher than the one that's active

             * OR

             * allow seize for requesting locality if no locality is

             * active

             */

            while ((TPM_TIS_IS_VALID_LOCTY(tis->active_locty) &&

                    locty > tis->active_locty) ||

                    !TPM_TIS_IS_VALID_LOCTY(tis->active_locty)) {

                bool higher_seize = FALSE;



                /* already a pending SEIZE ? */

                if ((tis->loc[locty].access & TPM_TIS_ACCESS_SEIZE)) {

                    break;

                }



                /* check for ongoing seize by a higher locality */

                for (l = locty + 1; l < TPM_TIS_NUM_LOCALITIES; l++) {

                    if ((tis->loc[l].access & TPM_TIS_ACCESS_SEIZE)) {

                        higher_seize = TRUE;

                        break;

                    }

                }



                if (higher_seize) {

                    break;

                }



                /* cancel any seize by a lower locality */

                for (l = 0; l < locty - 1; l++) {

                    tis->loc[l].access &= ~TPM_TIS_ACCESS_SEIZE;

                }



                tis->loc[locty].access |= TPM_TIS_ACCESS_SEIZE;

                DPRINTF("tpm_tis: TPM_TIS_ACCESS_SEIZE: "

                        "Locality %d seized from locality %d\n",

                        locty, tis->active_locty);

                DPRINTF("tpm_tis: TPM_TIS_ACCESS_SEIZE: Initiating abort.\n");

                set_new_locty = 0;

                tpm_tis_prep_abort(s, tis->active_locty, locty);

                break;

            }

        }



        if ((val & TPM_TIS_ACCESS_REQUEST_USE)) {

            if (tis->active_locty != locty) {

                if (TPM_TIS_IS_VALID_LOCTY(tis->active_locty)) {

                    tis->loc[locty].access |= TPM_TIS_ACCESS_REQUEST_USE;

                } else {

                    /* no locality active -> make this one active now */

                    active_locty = locty;

                }

            }

        }



        if (set_new_locty) {

            tpm_tis_new_active_locality(s, active_locty);

        }



        break;

    case TPM_TIS_REG_INT_ENABLE:

        if (tis->active_locty != locty) {

            break;

        }



        tis->loc[locty].inte &= mask;

        tis->loc[locty].inte |= (val & (TPM_TIS_INT_ENABLED |

                                        TPM_TIS_INT_POLARITY_MASK |

                                        TPM_TIS_INTERRUPTS_SUPPORTED));

        break;

    case TPM_TIS_REG_INT_VECTOR:

        /* hard wired -- ignore */

        break;

    case TPM_TIS_REG_INT_STATUS:

        if (tis->active_locty != locty) {

            break;

        }



        /* clearing of interrupt flags */

        if (((val & TPM_TIS_INTERRUPTS_SUPPORTED)) &&

            (tis->loc[locty].ints & TPM_TIS_INTERRUPTS_SUPPORTED)) {

            tis->loc[locty].ints &= ~val;

            if (tis->loc[locty].ints == 0) {

                qemu_irq_lower(tis->irq);

                DPRINTF("tpm_tis: Lowering IRQ\n");

            }

        }

        tis->loc[locty].ints &= ~(val & TPM_TIS_INTERRUPTS_SUPPORTED);

        break;

    case TPM_TIS_REG_STS:

        if (tis->active_locty != locty) {

            break;

        }



        val &= (TPM_TIS_STS_COMMAND_READY | TPM_TIS_STS_TPM_GO |

                TPM_TIS_STS_RESPONSE_RETRY);



        if (val == TPM_TIS_STS_COMMAND_READY) {

            switch (tis->loc[locty].state) {



            case TPM_TIS_STATE_READY:

                tis->loc[locty].w_offset = 0;

                tis->loc[locty].r_offset = 0;

            break;



            case TPM_TIS_STATE_IDLE:

                tis->loc[locty].sts = TPM_TIS_STS_COMMAND_READY;

                tis->loc[locty].state = TPM_TIS_STATE_READY;

                tpm_tis_raise_irq(s, locty, TPM_TIS_INT_COMMAND_READY);

            break;



            case TPM_TIS_STATE_EXECUTION:

            case TPM_TIS_STATE_RECEPTION:

                /* abort currently running command */

                DPRINTF("tpm_tis: %s: Initiating abort.\n",

                        __func__);

                tpm_tis_prep_abort(s, locty, locty);

            break;



            case TPM_TIS_STATE_COMPLETION:

                tis->loc[locty].w_offset = 0;

                tis->loc[locty].r_offset = 0;

                /* shortcut to ready state with C/R set */

                tis->loc[locty].state = TPM_TIS_STATE_READY;

                if (!(tis->loc[locty].sts & TPM_TIS_STS_COMMAND_READY)) {

                    tis->loc[locty].sts   = TPM_TIS_STS_COMMAND_READY;

                    tpm_tis_raise_irq(s, locty, TPM_TIS_INT_COMMAND_READY);

                }

                tis->loc[locty].sts &= ~(TPM_TIS_STS_DATA_AVAILABLE);

            break;



            }

        } else if (val == TPM_TIS_STS_TPM_GO) {

            switch (tis->loc[locty].state) {

            case TPM_TIS_STATE_RECEPTION:

                if ((tis->loc[locty].sts & TPM_TIS_STS_EXPECT) == 0) {

                    tpm_tis_tpm_send(s, locty);

                }

                break;

            default:

                /* ignore */

                break;

            }

        } else if (val == TPM_TIS_STS_RESPONSE_RETRY) {

            switch (tis->loc[locty].state) {

            case TPM_TIS_STATE_COMPLETION:

                tis->loc[locty].r_offset = 0;

                tis->loc[locty].sts = TPM_TIS_STS_VALID |

                                      TPM_TIS_STS_DATA_AVAILABLE;

                break;

            default:

                /* ignore */

                break;

            }

        }

        break;

    case TPM_TIS_REG_DATA_FIFO:

    case TPM_TIS_REG_DATA_XFIFO ... TPM_TIS_REG_DATA_XFIFO_END:

        /* data fifo */

        if (tis->active_locty != locty) {

            break;

        }



        if (tis->loc[locty].state == TPM_TIS_STATE_IDLE ||

            tis->loc[locty].state == TPM_TIS_STATE_EXECUTION ||

            tis->loc[locty].state == TPM_TIS_STATE_COMPLETION) {

            /* drop the byte */

        } else {

            DPRINTF("tpm_tis: Data to send to TPM: %08x (size=%d)\n",

                    val, size);

            if (tis->loc[locty].state == TPM_TIS_STATE_READY) {

                tis->loc[locty].state = TPM_TIS_STATE_RECEPTION;

                tis->loc[locty].sts = TPM_TIS_STS_EXPECT | TPM_TIS_STS_VALID;

            }



            val >>= shift;

            if (size > 4 - (addr & 0x3)) {

                /* prevent access beyond FIFO */

                size = 4 - (addr & 0x3);

            }



            while ((tis->loc[locty].sts & TPM_TIS_STS_EXPECT) && size > 0) {

                if (tis->loc[locty].w_offset < tis->loc[locty].w_buffer.size) {

                    tis->loc[locty].w_buffer.

                        buffer[tis->loc[locty].w_offset++] = (uint8_t)val;

                    val >>= 8;

                    size--;

                } else {

                    tis->loc[locty].sts = TPM_TIS_STS_VALID;

                }

            }



            /* check for complete packet */

            if (tis->loc[locty].w_offset > 5 &&

                (tis->loc[locty].sts & TPM_TIS_STS_EXPECT)) {

                /* we have a packet length - see if we have all of it */

#ifdef RAISE_STS_IRQ

                bool needIrq = !(tis->loc[locty].sts & TPM_TIS_STS_VALID);

#endif

                len = tpm_tis_get_size_from_buffer(&tis->loc[locty].w_buffer);

                if (len > tis->loc[locty].w_offset) {

                    tis->loc[locty].sts = TPM_TIS_STS_EXPECT |

                                          TPM_TIS_STS_VALID;

                } else {

                    /* packet complete */

                    tis->loc[locty].sts = TPM_TIS_STS_VALID;

                }

#ifdef RAISE_STS_IRQ

                if (needIrq) {

                    tpm_tis_raise_irq(s, locty, TPM_TIS_INT_STS_VALID);

                }

#endif

            }

        }

        break;

    }

}
