static void eeprom24c0x_write(int scl, int sda)

{

    if (eeprom.scl && scl && (eeprom.sda != sda)) {

        logout("%u: scl = %u->%u, sda = %u->%u i2c %s\n",

                eeprom.tick, eeprom.scl, scl, eeprom.sda, sda, sda ? "stop" : "start");

        if (!sda) {

            eeprom.tick = 1;

            eeprom.command = 0;

        }

    } else if (eeprom.tick == 0 && !eeprom.ack) {

        /* Waiting for start. */

        logout("%u: scl = %u->%u, sda = %u->%u wait for i2c start\n",

                eeprom.tick, eeprom.scl, scl, eeprom.sda, sda);

    } else if (!eeprom.scl && scl) {

        logout("%u: scl = %u->%u, sda = %u->%u trigger bit\n",

                eeprom.tick, eeprom.scl, scl, eeprom.sda, sda);

        if (eeprom.ack) {

            logout("\ti2c ack bit = 0\n");

            sda = 0;

            eeprom.ack = 0;

        } else if (eeprom.sda == sda) {

            uint8_t bit = (sda != 0);

            logout("\ti2c bit = %d\n", bit);

            if (eeprom.tick < 9) {

                eeprom.command <<= 1;

                eeprom.command += bit;

                eeprom.tick++;

                if (eeprom.tick == 9) {

                    logout("\tcommand 0x%04x, %s\n", eeprom.command, bit ? "read" : "write");

                    eeprom.ack = 1;

                }

            } else if (eeprom.tick < 17) {

                if (eeprom.command & 1) {

                    sda = ((eeprom.data & 0x80) != 0);

                }

                eeprom.address <<= 1;

                eeprom.address += bit;

                eeprom.tick++;

                eeprom.data <<= 1;

                if (eeprom.tick == 17) {

                    eeprom.data = eeprom.contents[eeprom.address];

                    logout("\taddress 0x%04x, data 0x%02x\n", eeprom.address, eeprom.data);

                    eeprom.ack = 1;

                    eeprom.tick = 0;

                }

            } else if (eeprom.tick >= 17) {

                sda = 0;

            }

        } else {

            logout("\tsda changed with raising scl\n");

        }

    } else {

        logout("%u: scl = %u->%u, sda = %u->%u\n", eeprom.tick, eeprom.scl, scl, eeprom.sda, sda);

    }

    eeprom.scl = scl;

    eeprom.sda = sda;

}
