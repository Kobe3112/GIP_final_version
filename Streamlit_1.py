import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
import numpy as np

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if os.path.exists("data.pkl"):
    try:
        with open("data.pkl", "rb") as f:
            geladen_dict = pickle.load(f)
            warmtepompen_simulatie_alles = geladen_dict.get("warmtepompen_simulatie", {})
            #drukverliezen = geladen_dict.get("drukverliezen",[])
            pompvermogen = geladen_dict.get("pompvermogen",[])
            st.session_state["WP7_8_9_checkbox"] = geladen_dict.get("WP7_8_9_checkbox", False)

            if not st.session_state.get("WP7_8_9_checkbox", False):
                del warmtepompen_simulatie_alles["WP7"]
                del warmtepompen_simulatie_alles["WP8"]
                del warmtepompen_simulatie_alles["WP9"]
            # Mapping voor het hernoemen van keys
            rename_inner_keys = {
                'Q_building': 'Q HP [W]',
                'Q_boiler': 'Q boiler [W]',
                'T_building': 'T HP [°C]',
                'T_boiler': 'T boiler [°C]',
                'T_in_source': 'T in [°C]',
                'T_out_source': 'T out [°C]',
                'm': 'm [kg/s]',
                'delta_T': 'ΔT [°C]',
                'COP': 'COP [-]',
                'lengte': 'length pipe [m]',
                'snelheid': 'v fluid [m/s]',
                'percentage': 'percentage [%]',
                'P_compressor': 'P compressor [W]',
                'T_req_building': 'T demand building [°C]',
                'Q_req_building': 'Q demand building [W]',
                'diameter': 'diameter [cm]'
            }

            # Keys die je binnen elke subdict wil verwijderen
            keys_to_remove = {'selected_model'}

            # Nieuwe dictionary opbouwen
            warmtepompen_simulatie = {
                wp: {
                    rename_inner_keys.get(k, k): v
                    for k, v in data.items()
                    if k not in keys_to_remove
                }
                for wp, data in warmtepompen_simulatie_alles.items()
            }
            for wp_data in warmtepompen_simulatie.values():
                if 'percentage [%]' in wp_data:
                    wp_data['percentage [%]'] = [value * 100 for value in wp_data['percentage [%]']]
                if 'diameter [cm]' in wp_data:
                    wp_data['diameter [cm]'] = [value * 100 for value in wp_data['diameter [cm]']]


        st.success("Sessie geladen van lokale 'data.pkl'.")
    except Exception as e:
        st.error(f"Fout bij lokale fallback: {e}")


aantal_kolommen_grafieken = 3

if "behouden_grafieken" not in st.session_state:
    st.session_state.behouden_grafieken = []

# Tabs maken
tab1, tab2 = st.tabs(["Standaard grafiek", "Sommaties"])

# **Eerste tab: Standaard grafiek**
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        wp_ids = list(warmtepompen_simulatie.keys())
        if "selected_wps" not in st.session_state:
            st.session_state.selected_wps = ['WP1']

        def update_selected_wps():
            st.session_state.selected_wps = st.session_state.wp_multiselect_tab1

        if st.button("Selecteer alle warmtepompen", key="selecteer_alle_wp_tab1"):
            st.session_state.selected_wps = wp_ids
            st.session_state.wp_multiselect_tab1 = wp_ids

        # Filter alleen de warmtepompen die bestaan in de data
        geldige_defaults = [wp for wp in st.session_state.selected_wps if wp in wp_ids]

        selected_wps = st.multiselect(
            "Selecteer warmtepompen",
            wp_ids,
            default=geldige_defaults,
            key="wp_multiselect_tab1",
            on_change=update_selected_wps
        )
        if selected_wps:
            voorbeeld_wp = selected_wps[0]
            parameter_lijst = list(warmtepompen_simulatie[voorbeeld_wp].keys())
            parameters = st.multiselect("Selecteer parameter(s)", parameter_lijst, key="parameters_tab1")
        else:
            st.warning("Selecteer minstens één warmtepomp om parameters te kunnen kiezen.")

        x_as_optie = st.selectbox(
            "Kies x-as tijdsindeling:",
            ["Maandstarten", "Max 6 labels", "Kwartaal", "Custom"],
            index=0,
            key="x_as_optie_tab1"
        )

    # Tijdstempel genereren (1 waarde per uur in 2022)
    tijdstempel = pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="H")

    if parameters and selected_wps:
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for parameter in parameters:
                for wp in selected_wps:
                    y = np.array(warmtepompen_simulatie[wp][parameter], dtype=float)
                    t = tijdstempel[:len(y)]
                    y[y == 0] = np.nan
                    ax.plot(t, y, linewidth=0.2, label=f"{wp} - {parameter}")

            ax.set_xlim(tijdstempel[0], tijdstempel[-1])
            ax.set_xlabel("Tijd")
            ax.set_ylabel("Waarde")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')

            # x-as formattering
            if x_as_optie == "Maandstarten":
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_as_optie == "Max 6 labels":
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
            elif x_as_optie == "Kwartaal":
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_as_optie == "Custom":
                custom_ticks = pd.to_datetime(["2022-01-01", "2022-07-01", "2022-12-31"])
                ax.set_xticks(custom_ticks)
                ax.set_xticklabels(custom_ticks.strftime("%d-%m"))

            st.pyplot(fig)

    if st.button("Behoud grafiek", key="behoud_grafiek_tab1") and parameters:
        st.session_state.behouden_grafieken.append({'wps': selected_wps.copy(), 'parameters': parameters.copy(), 'x_as': x_as_optie})
        st.rerun()

# **Tweede tab: Sommaties grafiek**
with tab2:
    col1, col2 = st.columns([1, 3])
    with col1:
        wp_ids = list(warmtepompen_simulatie.keys())
        if "selected_wps" not in st.session_state:
            st.session_state.selected_wps = ['WP1']

        def update_selected_wps():
            st.session_state.selected_wps = st.session_state.wp_multiselect_tab2

        selected_wps = st.multiselect("Selecteer warmtepompen", wp_ids, default=st.session_state.selected_wps, key="wp_multiselect_tab2", on_change=update_selected_wps)

        if selected_wps:
            voorbeeld_wp = selected_wps[0]
            parameter_lijst = list(warmtepompen_simulatie[voorbeeld_wp].keys())
            som_param = st.multiselect("Sommeer deze parameter(s) over alle geselecteerde warmtepompen", parameter_lijst, key="som_param_tab2")
        else:
            st.warning("Selecteer minstens één warmtepomp om parameters te kunnen kiezen.")

        x_as_optie = st.selectbox(
            "Kies x-as tijdsindeling:",
            ["Maandstarten", "Max 6 labels", "Kwartaal", "Custom"],
            index=0,
            key="x_as_optie_tab2"
        )

        toon_individuele = st.checkbox("Toon individuele lijnen?", value=True, key="toon_individuele_tab2")

    # Tijdstempel genereren (1 waarde per uur in 2022)
    tijdstempel = pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="H")

    if som_param and selected_wps:
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))

            # Sommeer alle geselecteerde parameters binnen elke warmtepomp en tel ze bij elkaar op
            totaal = pd.Series(0, index=range(len(tijdstempel)))

            for wp in selected_wps:
                som_wp = pd.Series(0, index=range(len(tijdstempel)))
                for param in som_param:
                    som_wp += pd.Series(warmtepompen_simulatie[wp][param])
                totaal += som_wp

            # Plot totaal
            ax.plot(tijdstempel[:len(totaal)], totaal, label=f"Som({', '.join(som_param)})", linewidth=0.5)

            # Toon indien aangevinkt: individuele lijnen per parameter per WP
            if toon_individuele:
                for wp in selected_wps:
                    for param in som_param:
                        y = np.array(warmtepompen_simulatie[wp][parameter], dtype=float)
                        t = tijdstempel[:len(y)]
                        y[y == 0] = np.nan
                        ax.plot(t, y, linewidth=0.2, label=f"{wp} - {parameter}")
            ax.set_xlim(tijdstempel[0], tijdstempel[-1])
            ax.set_xlabel("Tijd")
            ax.set_ylabel("Waarde")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')

            # x-as formattering
            if x_as_optie == "Maandstarten":
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_as_optie == "Max 6 labels":
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
            elif x_as_optie == "Kwartaal":
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_as_optie == "Custom":
                custom_ticks = pd.to_datetime(["2022-01-01", "2022-07-01", "2022-12-31"])
                ax.set_xticks(custom_ticks)
                ax.set_xticklabels(custom_ticks.strftime("%d-%m"))

            st.pyplot(fig)

    if st.button("Behoud grafiek", key="behoud_grafiek_tab2") and som_param:
        st.session_state.behouden_grafieken.append({
            'wps': selected_wps.copy(),
            'parameters': som_param.copy(),
            'x_as': x_as_optie,
            'toon_individuele': toon_individuele,
            'is_som': True
        })

        st.rerun()

if st.session_state.behouden_grafieken:
    st.subheader("Behouden grafieken")
    cols = st.columns(aantal_kolommen_grafieken)
    for idx, grafiek in enumerate(st.session_state.behouden_grafieken):
        col = cols[idx % aantal_kolommen_grafieken]
        with col:
            wps = grafiek['wps']
            params = grafiek['parameters']
            x_optie = grafiek.get('x_as', 'Maandstarten')  # Standaard naar 'Maandstarten'

            # Titel boven grafiek
            param_str = ", ".join(params)
            wp_str = ", ".join(wps)
            st.markdown(f"#### {param_str} voor {wp_str}")

            fig, ax = plt.subplots(figsize=(5, 3))

            is_som = grafiek.get('is_som', False)
            toon_individuele = grafiek.get('toon_individuele', True)

            if is_som:
                totaal = pd.Series(0, index=range(len(tijdstempel)))
                for wp in wps:
                    som_wp = pd.Series(0, index=range(len(tijdstempel)))
                    for param in params:
                        som_wp += pd.Series(warmtepompen_simulatie[wp][param])
                    totaal += som_wp

                ax.plot(tijdstempel[:len(totaal)], totaal, label=f"Som({', '.join(params)})", linewidth=0.5)

                if toon_individuele:
                    for wp in wps:
                        for param in params:
                            y = np.array(warmtepompen_simulatie[wp][parameter], dtype=float)
                            t = tijdstempel[:len(y)]
                            y[y == 0] = np.nan
                            ax.plot(t, y, linewidth=0.2, label=f"{wp} - {parameter}")
            else:
                for param in params:
                    for wp in wps:
                        y = np.array(warmtepompen_simulatie[wp][parameter], dtype=float)
                        t = tijdstempel[:len(y)]
                        y[y == 0] = np.nan
                        ax.plot(t, y, linewidth=0.2, label=f"{wp} - {parameter}")

            ax.set_xlim(tijdstempel[0], tijdstempel[-1])
            ax.set_xlabel("Tijd")

            # Dynamisch y-as label
            if len(params) == 1:
                ax.set_ylabel(params[0])
            else:
                ax.set_ylabel("Waarde")

            # Titel in de grafiek zelf (optioneel)
            if len(params) == 1:
                ax.set_title(f"{params[0]} voor geselecteerde warmtepompen")
            else:
                ax.set_title("Meerdere parameters")

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')

            # x-as formattering
            if x_optie == "Maandstarten":
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_optie == "Max 6 labels":
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
            elif x_optie == "Kwartaal":
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            elif x_optie == "Custom":
                custom_ticks = pd.to_datetime(["2022-01-01", "2022-07-01", "2022-12-31"])
                ax.set_xticks(custom_ticks)
                ax.set_xticklabels(custom_ticks.strftime("%d-%m"))

            st.pyplot(fig)

            data = {f"{wp} - {param}": warmtepompen_simulatie[wp][param] for param in params for wp in wps}
            df = pd.DataFrame(data)

            if is_som:
                # Som over parameters en/of warmtepompen
                som_series = pd.Series(0, index=range(len(tijdstempel)))
                for wp in wps:
                    for param in params:
                        som_series += pd.Series(warmtepompen_simulatie[wp][param])
                df["Som"] = som_series

            csv = df.to_csv(index=False, sep=';', decimal=',', float_format='%.2f')
            st.download_button(label="Download CSV",data=csv,file_name=f"grafiek_{idx + 1}.csv",mime="text/csv",key=f"dl_{idx}")

            if st.button("🗑️ Verwijder", key=f"verwijder_{idx}"):
                st.session_state.behouden_grafieken.pop(idx)
                st.rerun()
    st.markdown("---")
    if st.button("🗑️ Verwijder alle behouden grafieken", key="reset_all"):
        st.session_state.behouden_grafieken = []
        st.rerun()
st.markdown("---")


col1, col2 = st.columns([1, 3])
with col1:
    pompvermogen_check = st.checkbox("Toon pompvermogen")
with col2:
    if pompvermogen_check:
        tijdstempel_pompvermogen = pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="H")
        # Kies dezelfde x-as optie als je andere grafieken
        x_optie = "Maandstarten"  # of "Max 6 labels", "Kwartaal", "Custom"

        st.markdown("#### Pompvermogen over het jaar")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(tijdstempel_pompvermogen, pompvermogen, label="Pompvermogen", linewidth=0.5)

        ax.set_xlim(tijdstempel_pompvermogen[0], tijdstempel_pompvermogen[-1])
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Pompvermogen (W)")
        ax.set_title("Pompvermogen over tijd")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')

        # === X-as formattering hergebruikt ===
        if x_optie == "Maandstarten":
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        elif x_optie == "Max 6 labels":
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        elif x_optie == "Kwartaal":
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        elif x_optie == "Custom":
            custom_ticks = pd.to_datetime(["2024-01-01", "2024-07-01", "2024-12-31"])
            ax.set_xticks(custom_ticks)
            ax.set_xticklabels(custom_ticks.strftime("%d-%m"))

        st.pyplot(fig)

        # Optionele download
        df_pompvermogen = pd.DataFrame({
            "Pompvermogen": pompvermogen
        })
        csv_pomp = df_pompvermogen.to_csv(index=False, sep=';', decimal=',', float_format='%.2f')
        st.download_button(label="Download CSV", data=csv_pomp, file_name="pompvermogen.csv", mime="text/csv",key="pomp")


st.markdown("---")
st.subheader("Sessie opslaan")

default_filename = "mijn_sessie"
filename_input = st.text_input("Bestandsnaam (zonder extensie)", default_filename)

if st.button("📦 Sla sessie op"):
    save_dict = {
        "warmtepompen_simulatie": warmtepompen_simulatie,
        "checkbox_status": st.session_state.get("WP7_8_9_checkbox", False),
        "behouden_grafieken": st.session_state.get("behouden_grafieken", []),
        #"drukverliezen": drukverliezen,
        "pompvermogen": pompvermogen
    }

    # Zorg dat bestandsnaam veilig is
    safe_filename = f"{filename_input.strip() or default_filename}.pkl"
    with open(safe_filename, "wb") as f:
        pickle.dump(save_dict, f)
    st.success(f"Sessie opgeslagen als {safe_filename}")
