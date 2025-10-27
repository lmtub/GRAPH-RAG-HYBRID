void acpi_pm1_cnt_write(ACPIREGS *ar, uint16_t val)

{

    ar->pm1.cnt.cnt = val & ~(ACPI_BITMASK_SLEEP_ENABLE);



    if (val & ACPI_BITMASK_SLEEP_ENABLE) {

        /* change suspend type */

        uint16_t sus_typ = (val >> 10) & 7;

        switch(sus_typ) {

        case 0: /* soft power off */

            qemu_system_shutdown_request();

            break;

        case 1:

            /* ACPI_BITMASK_WAKE_STATUS should be set on resume.

               Pretend that resume was caused by power button */

            ar->pm1.evt.sts |=

                (ACPI_BITMASK_WAKE_STATUS | ACPI_BITMASK_POWER_BUTTON_STATUS);

            qemu_system_reset_request();

            qemu_irq_raise(ar->pm1.cnt.cmos_s3);

        default:

            break;

        }

    }

}
