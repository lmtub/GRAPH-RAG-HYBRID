static void set_year_20xx(void)
{
    /* Set BCD mode */
    cmos_write(RTC_REG_B, cmos_read(RTC_REG_B) & ~REG_B_DM);
    cmos_write(RTC_REG_A, 0x76);
    cmos_write(RTC_YEAR, 0x11);
    cmos_write(RTC_CENTURY, 0x20);
    cmos_write(RTC_MONTH, 0x02);
    cmos_write(RTC_DAY_OF_MONTH, 0x02);
    cmos_write(RTC_HOURS, 0x02);
    cmos_write(RTC_MINUTES, 0x04);
    cmos_write(RTC_SECONDS, 0x58);
    cmos_write(RTC_REG_A, 0x26);
    g_assert_cmpint(cmos_read(RTC_HOURS), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MINUTES), ==, 0x04);
    g_assert_cmpint(cmos_read(RTC_SECONDS), >=, 0x58);
    g_assert_cmpint(cmos_read(RTC_DAY_OF_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_YEAR), ==, 0x11);
    g_assert_cmpint(cmos_read(RTC_CENTURY), ==, 0x20);
    /* Set a date in 2080 to ensure there is no year-2038 overflow.  */
    cmos_write(RTC_REG_A, 0x76);
    cmos_write(RTC_YEAR, 0x80);
    cmos_write(RTC_REG_A, 0x26);
    g_assert_cmpint(cmos_read(RTC_HOURS), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MINUTES), ==, 0x04);
    g_assert_cmpint(cmos_read(RTC_SECONDS), >=, 0x58);
    g_assert_cmpint(cmos_read(RTC_DAY_OF_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_YEAR), ==, 0x80);
    g_assert_cmpint(cmos_read(RTC_CENTURY), ==, 0x20);
    cmos_write(RTC_REG_A, 0x76);
    cmos_write(RTC_YEAR, 0x11);
    cmos_write(RTC_REG_A, 0x26);
    g_assert_cmpint(cmos_read(RTC_HOURS), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MINUTES), ==, 0x04);
    g_assert_cmpint(cmos_read(RTC_SECONDS), >=, 0x58);
    g_assert_cmpint(cmos_read(RTC_DAY_OF_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_MONTH), ==, 0x02);
    g_assert_cmpint(cmos_read(RTC_YEAR), ==, 0x11);
    g_assert_cmpint(cmos_read(RTC_CENTURY), ==, 0x20);