void acpi_pm1_cnt_reset(ACPIREGS *ar)

{

    ar->pm1.cnt.cnt = 0;

    if (ar->pm1.cnt.cmos_s3) {

        qemu_irq_lower(ar->pm1.cnt.cmos_s3);

    }

}
