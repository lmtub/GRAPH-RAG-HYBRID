static void exynos4210_i2c_write(void *opaque, target_phys_addr_t offset,

                              uint64_t value, unsigned size)

{

    Exynos4210I2CState *s = (Exynos4210I2CState *)opaque;

    uint8_t v = value & 0xff;



    DPRINT("write %s [0x%02x] <- 0x%02x\n", exynos4_i2c_get_regname(offset),

            (unsigned int)offset, v);



    switch (offset) {

    case I2CCON_ADDR:

        s->i2ccon = (v & ~I2CCON_INT_PEND) | (s->i2ccon & I2CCON_INT_PEND);

        if ((s->i2ccon & I2CCON_INT_PEND) && !(v & I2CCON_INT_PEND)) {

            s->i2ccon &= ~I2CCON_INT_PEND;

            qemu_irq_lower(s->irq);

            if (!(s->i2ccon & I2CCON_INTRS_EN)) {

                s->i2cstat &= ~I2CSTAT_START_BUSY;

            }



            if (s->i2cstat & I2CSTAT_START_BUSY) {

                if (s->scl_free) {

                    if (EXYNOS4_I2C_MODE(s->i2cstat) == I2CMODE_MASTER_Tx) {

                        exynos4210_i2c_data_send(s);

                    } else if (EXYNOS4_I2C_MODE(s->i2cstat) ==

                            I2CMODE_MASTER_Rx) {

                        exynos4210_i2c_data_receive(s);

                    }

                } else {

                    s->i2ccon |= I2CCON_INT_PEND;

                    qemu_irq_raise(s->irq);

                }

            }

        }

        break;

    case I2CSTAT_ADDR:

        s->i2cstat =

                (s->i2cstat & I2CSTAT_START_BUSY) | (v & ~I2CSTAT_START_BUSY);



        if (!(s->i2cstat & I2CSTAT_OUTPUT_EN)) {

            s->i2cstat &= ~I2CSTAT_START_BUSY;

            s->scl_free = true;

            qemu_irq_lower(s->irq);

            break;

        }



        /* Nothing to do if in i2c slave mode */

        if (!I2C_IN_MASTER_MODE(s->i2cstat)) {

            break;

        }



        if (v & I2CSTAT_START_BUSY) {

            s->i2cstat &= ~I2CSTAT_LAST_BIT;

            s->i2cstat |= I2CSTAT_START_BUSY;    /* Line is busy */

            s->scl_free = false;



            /* Generate start bit and send slave address */

            if (i2c_start_transfer(s->bus, s->i2cds >> 1, s->i2cds & 0x1) &&

                    (s->i2ccon & I2CCON_ACK_GEN)) {

                s->i2cstat |= I2CSTAT_LAST_BIT;

            } else if (EXYNOS4_I2C_MODE(s->i2cstat) == I2CMODE_MASTER_Rx) {

                exynos4210_i2c_data_receive(s);

            }

            exynos4210_i2c_raise_interrupt(s);

        } else {

            i2c_end_transfer(s->bus);

            if (!(s->i2ccon & I2CCON_INT_PEND)) {

                s->i2cstat &= ~I2CSTAT_START_BUSY;

            }

            s->scl_free = true;

        }

        break;

    case I2CADD_ADDR:

        if ((s->i2cstat & I2CSTAT_OUTPUT_EN) == 0) {

            s->i2cadd = v;

        }

        break;

    case I2CDS_ADDR:

        if (s->i2cstat & I2CSTAT_OUTPUT_EN) {

            s->i2cds = v;

            s->scl_free = true;

            if (EXYNOS4_I2C_MODE(s->i2cstat) == I2CMODE_MASTER_Tx &&

                    (s->i2cstat & I2CSTAT_START_BUSY) &&

                    !(s->i2ccon & I2CCON_INT_PEND)) {

                exynos4210_i2c_data_send(s);

            }

        }

        break;

    case I2CLC_ADDR:

        s->i2clc = v;

        break;

    default:

        DPRINT("ERROR: Bad write offset 0x%x\n", (unsigned int)offset);

        break;

    }

}
