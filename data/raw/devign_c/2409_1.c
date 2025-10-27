static void pnv_chip_power8_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PnvChipClass *k = PNV_CHIP_CLASS(klass);



    k->cpu_model = "POWER8";

    k->chip_type = PNV_CHIP_POWER8;

    k->chip_cfam_id = 0x220ea04980000000ull; /* P8 Venice DD2.0 */

    k->cores_mask = POWER8_CORE_MASK;

    k->core_pir = pnv_chip_core_pir_p8;


    dc->desc = "PowerNV Chip POWER8";

}