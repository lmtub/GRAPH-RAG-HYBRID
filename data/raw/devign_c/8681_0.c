void omap_rfbi_attach(struct omap_dss_s *s, int cs, struct rfbi_chip_s *chip)

{

    if (cs < 0 || cs > 1)

        hw_error("%s: wrong CS %i\n", __FUNCTION__, cs);

    s->rfbi.chip[cs] = chip;

}
