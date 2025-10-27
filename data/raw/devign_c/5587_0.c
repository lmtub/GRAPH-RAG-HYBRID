static uint64_t megasas_fw_time(void)

{

    struct tm curtime;

    uint64_t bcd_time;



    qemu_get_timedate(&curtime, 0);

    bcd_time = ((uint64_t)curtime.tm_sec & 0xff) << 48 |

        ((uint64_t)curtime.tm_min & 0xff)  << 40 |

        ((uint64_t)curtime.tm_hour & 0xff) << 32 |

        ((uint64_t)curtime.tm_mday & 0xff) << 24 |

        ((uint64_t)curtime.tm_mon & 0xff)  << 16 |

        ((uint64_t)(curtime.tm_year + 1900) & 0xffff);



    return bcd_time;

}
