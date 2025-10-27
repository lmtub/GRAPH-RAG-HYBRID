void omap_uwire_attach(struct omap_uwire_s *s,

                uWireSlave *slave, int chipselect)

{

    if (chipselect < 0 || chipselect > 3) {

        fprintf(stderr, "%s: Bad chipselect %i\n", __FUNCTION__, chipselect);

        exit(-1);

    }



    s->chip[chipselect] = slave;

}
