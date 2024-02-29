import pandas as pd
import numpy as np
import sys

from pydantic import ValidationError

from config import load_config, OutputDensity


MASS_FRACTION_INDEX = 3
T_INDEX = "Temperature"
P_INDEX = "Pressure"
OUTPUT_FILE = "output.txt"
DATA_FILE = "data.csv"
CONFIG_PATH = "a2e.yaml"


def main():
    try:
        c = load_config(CONFIG_PATH)
    except ValidationError as e:
        print("ERROR: configuration file has the following errors:")
        for e in e.errors():
            loc = [str(i) for i in e["loc"]]
            print(f"\t{'.'.join(loc)} -> {e['msg']}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, header=0, index_col=0)

    try:
        check_units(df)
    except Exception as e:
        exit_with_err(e.__str__())

    indexes = df.index.to_numpy().astype(str)
    chems = indexes[MASS_FRACTION_INDEX + 1 :]

    column_names = df.columns.to_numpy(dtype=str)
    streams = column_names[
        np.vectorize(lambda c: c.startswith("S") and c != "SULFUR")(column_names)
    ]

    for chem in chems:
        if chem not in c.ees_aliases:
            exit_with_err(f"ees alias for chemical {chem} does not exist")

    f = open(OUTPUT_FILE, "w")

    def write_line(line: str):
        f.write(line + "\n")

    fp = float_printer(c.max_decimals)
    for chem in chems:
        ch = chem
        if chem in c.short_aliases:
            ch = c.short_aliases[chem]
        ees_chem = c.ees_aliases[chem]
        write_line(f"h0_{ch} = Enthalpy({ees_chem}, T=298.15, P=101.325)")
        write_line(f"s0_{ch} = Entropy({ees_chem}, T=298.15, P=101.325)")
    write_line("")

    should_print_chem_space = c.output_density == OutputDensity.WIDE
    should_print_stream_space = c.output_density != OutputDensity.DENSE

    for stream in streams:
        stream_enthalpies = []
        stream_entropies = []
        stream_exergies = []
        if c.comments:
            write_line(f'"{stream}"')
        for chem in chems:
            mf = df[stream][chem]
            if mf == 0.0:
                continue
            t = df[stream][T_INDEX]
            # pressure for the mass fraction
            p = df[stream][P_INDEX] * mf
            ees_chem = c.ees_aliases[chem]
            ch = chem
            if chem in c.short_aliases:
                ch = c.short_aliases[chem]
            sc_suffix, _ = stream_chem_suffix(prefix=c.prefix, raw=stream, chem=ch)

            enthalpy_name = f"h{sc_suffix}"
            entropy_name = f"s{sc_suffix}"
            exergy_name = f"EX{sc_suffix}"
            t_name = f"t{sc_suffix}"
            p_name = f"p{sc_suffix}"
            write_line(f"{t_name} = {fp(t)}")
            write_line(f"{p_name} = {fp(p)}")
            write_line(
                f"{enthalpy_name} = Enthalpy({ees_chem}, T={t_name}, P={p_name})"
            )
            stream_enthalpies.append(enthalpy_name)
            write_line(f"{entropy_name} = Entropy({ees_chem}, T={t_name}, P={p_name})")
            stream_entropies.append(entropy_name)
            write_line(
                f"{exergy_name} = {fp(mf)} * (({enthalpy_name} - h0_{c}) - 298.15 * ({entropy_name} - s0_{c}))"
            )
            stream_exergies.append(exergy_name)
            if should_print_chem_space:
                write_line("")

        t = df[stream][T_INDEX]
        p = df[stream][P_INDEX]
        stream_num = get_stream_num(prefix=c.prefix, raw=stream)
        s_suffix = f"[{stream_num}]" if stream_num != -1 else f"_{stream}"
        write_line(f"t{s_suffix} = {fp(t)}")
        write_line(f"p{s_suffix} = {fp(p)}")
        write_line(f"h{s_suffix} = {' + '.join(stream_enthalpies)}")
        write_line(f"s{s_suffix} = {' + '.join(stream_entropies)}")
        write_line(f"EX{s_suffix} = {' + '.join(stream_exergies)}")
        if should_print_stream_space:
            write_line("")

    f.close()


def check_units(df: pd.DataFrame):
    temp_unit = df["Units"][T_INDEX]
    if temp_unit != "K":
        raise ValueError(
            f"Temperature unit should be 'K' (Kelvin), not '{temp_unit}' re-export the data from Aspen with the appropriate unit"
        )

    pressure_unit = df["Units"][P_INDEX]
    if pressure_unit != "KPa":
        raise ValueError(
            f"Pressure unit should be 'KPa' (KiloPascal), not '{pressure_unit}' re-export the data from Aspen with the appropriate unit"
        )


def stream_chem_suffix(*, prefix: str, raw: str, chem: str) -> tuple[str, bool]:
    stream_num = get_stream_num(prefix=prefix, raw=raw)
    if stream_num != -1:
        return f"_{chem}[{stream_num}]", True
    return f"_{chem}_{raw}", False


def get_stream_num(*, prefix: str, raw: str) -> int:
    if not raw.startswith(prefix):
        return -1
    stream_num = raw[len(prefix) :]
    if stream_num.isdigit():
        return int(stream_num)
    return -1


def exit_with_err(err: str):
    print(f"ERROR: {err}")
    sys.exit(1)


def float_printer(decimals: int):
    def printer(f: float):
        res = f"{{:.{decimals}f}}".format(f).rstrip("0")
        if res[-1] == ".":
            return res[:-1]
        return res

    return printer


if __name__ == "__main__":
    main()
