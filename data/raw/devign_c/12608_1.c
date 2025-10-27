static const char *scsi_command_name(uint8_t cmd)

{

    static const char *names[] = {

        [ TEST_UNIT_READY          ] = "TEST_UNIT_READY",

        [ REQUEST_SENSE            ] = "REQUEST_SENSE",

        [ FORMAT_UNIT              ] = "FORMAT_UNIT",

        [ READ_BLOCK_LIMITS        ] = "READ_BLOCK_LIMITS",

        [ REASSIGN_BLOCKS          ] = "REASSIGN_BLOCKS",

        [ READ_6                   ] = "READ_6",

        [ WRITE_6                  ] = "WRITE_6",

        [ SEEK_6                   ] = "SEEK_6",

        [ READ_REVERSE             ] = "READ_REVERSE",

        [ WRITE_FILEMARKS          ] = "WRITE_FILEMARKS",

        [ SPACE                    ] = "SPACE",

        [ INQUIRY                  ] = "INQUIRY",

        [ RECOVER_BUFFERED_DATA    ] = "RECOVER_BUFFERED_DATA",

        [ MAINTENANCE_IN           ] = "MAINTENANCE_IN",

        [ MAINTENANCE_OUT          ] = "MAINTENANCE_OUT",

        [ MODE_SELECT              ] = "MODE_SELECT",

        [ RESERVE                  ] = "RESERVE",

        [ RELEASE                  ] = "RELEASE",

        [ COPY                     ] = "COPY",

        [ ERASE                    ] = "ERASE",

        [ MODE_SENSE               ] = "MODE_SENSE",

        [ START_STOP               ] = "START_STOP",

        [ RECEIVE_DIAGNOSTIC       ] = "RECEIVE_DIAGNOSTIC",

        [ SEND_DIAGNOSTIC          ] = "SEND_DIAGNOSTIC",

        [ ALLOW_MEDIUM_REMOVAL     ] = "ALLOW_MEDIUM_REMOVAL",



        [ READ_CAPACITY            ] = "READ_CAPACITY",

        [ READ_10                  ] = "READ_10",

        [ WRITE_10                 ] = "WRITE_10",

        [ SEEK_10                  ] = "SEEK_10",

        [ WRITE_VERIFY             ] = "WRITE_VERIFY",

        [ VERIFY                   ] = "VERIFY",

        [ SEARCH_HIGH              ] = "SEARCH_HIGH",

        [ SEARCH_EQUAL             ] = "SEARCH_EQUAL",

        [ SEARCH_LOW               ] = "SEARCH_LOW",

        [ SET_LIMITS               ] = "SET_LIMITS",

        [ PRE_FETCH                ] = "PRE_FETCH",

        /* READ_POSITION and PRE_FETCH use the same operation code */

        [ SYNCHRONIZE_CACHE        ] = "SYNCHRONIZE_CACHE",

        [ LOCK_UNLOCK_CACHE        ] = "LOCK_UNLOCK_CACHE",

        [ READ_DEFECT_DATA         ] = "READ_DEFECT_DATA",

        [ MEDIUM_SCAN              ] = "MEDIUM_SCAN",

        [ COMPARE                  ] = "COMPARE",

        [ COPY_VERIFY              ] = "COPY_VERIFY",

        [ WRITE_BUFFER             ] = "WRITE_BUFFER",

        [ READ_BUFFER              ] = "READ_BUFFER",

        [ UPDATE_BLOCK             ] = "UPDATE_BLOCK",

        [ READ_LONG                ] = "READ_LONG",

        [ WRITE_LONG               ] = "WRITE_LONG",

        [ CHANGE_DEFINITION        ] = "CHANGE_DEFINITION",

        [ WRITE_SAME               ] = "WRITE_SAME",

        [ READ_TOC                 ] = "READ_TOC",

        [ LOG_SELECT               ] = "LOG_SELECT",

        [ LOG_SENSE                ] = "LOG_SENSE",

        [ MODE_SELECT_10           ] = "MODE_SELECT_10",

        [ RESERVE_10               ] = "RESERVE_10",

        [ RELEASE_10               ] = "RELEASE_10",

        [ MODE_SENSE_10            ] = "MODE_SENSE_10",

        [ PERSISTENT_RESERVE_IN    ] = "PERSISTENT_RESERVE_IN",

        [ PERSISTENT_RESERVE_OUT   ] = "PERSISTENT_RESERVE_OUT",

        [ MOVE_MEDIUM              ] = "MOVE_MEDIUM",

        [ READ_12                  ] = "READ_12",

        [ WRITE_12                 ] = "WRITE_12",

        [ WRITE_VERIFY_12          ] = "WRITE_VERIFY_12",

        [ SEARCH_HIGH_12           ] = "SEARCH_HIGH_12",

        [ SEARCH_EQUAL_12          ] = "SEARCH_EQUAL_12",

        [ SEARCH_LOW_12            ] = "SEARCH_LOW_12",

        [ READ_ELEMENT_STATUS      ] = "READ_ELEMENT_STATUS",

        [ SEND_VOLUME_TAG          ] = "SEND_VOLUME_TAG",

        [ WRITE_LONG_2             ] = "WRITE_LONG_2",



        [ REPORT_DENSITY_SUPPORT   ] = "REPORT_DENSITY_SUPPORT",

        [ GET_CONFIGURATION        ] = "GET_CONFIGURATION",

        [ READ_16                  ] = "READ_16",

        [ WRITE_16                 ] = "WRITE_16",

        [ WRITE_VERIFY_16          ] = "WRITE_VERIFY_16",

        [ SERVICE_ACTION_IN        ] = "SERVICE_ACTION_IN",

        [ REPORT_LUNS              ] = "REPORT_LUNS",

        [ LOAD_UNLOAD              ] = "LOAD_UNLOAD",

        [ SET_CD_SPEED             ] = "SET_CD_SPEED",

        [ BLANK                    ] = "BLANK",

    };



    if (cmd >= ARRAY_SIZE(names) || names[cmd] == NULL)

        return "*UNKNOWN*";

    return names[cmd];

}
