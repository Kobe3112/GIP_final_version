import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygfunction as gt
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.colors as mcolors
from scipy.linalg import lstsq
import sys


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;  # Pas deze waarde aan naar de gewenste breedte
        margin: 0 auto;  # Centreer de inhoud
    }
    </style>
    """,
    unsafe_allow_html=True
)

warmtepompen = defaultdict(dict)
HP_data = defaultdict(dict)
leidingen = defaultdict(dict)
warmtepompen_kopie = defaultdict(dict)
warmtepompen_state = defaultdict(dict)
k = 1
#################################################
###### HIER KOMEN ALLE FUNCTIES #################
#################################################
def bereken_diameter(WP,snelheid):
    V_dot_WP = warmtepompen[WP]['m']/dichtheid_fluid_backbone
    opp = V_dot_WP/snelheid
    r = math.sqrt(opp/math.pi)
    return 2*r
def bereken_diameter_pipe(leiding,snelheid,temp):
    V_dot_leiding = leidingen[leiding][temp]['m'] / dichtheid_fluid_backbone
    if V_dot_leiding < 0:
        st.write("m gaat onder 0  🙁")
        V_dot_leiding = 0
    opp = V_dot_leiding / snelheid
    r = math.sqrt(opp / math.pi)
    return 2 * r
def bereken_snelheid(WP,diameter):
    r = diameter/2
    opp = math.pi*r*r
    V_dot_WP = warmtepompen[WP]['m']/dichtheid_fluid_backbone
    v = V_dot_WP/opp
    return v
def bereken_snelheid_pipe(leiding,diameter,temp):
    r = diameter / 2
    opp = math.pi * r * r
    V_dot_leiding = leidingen[leiding][temp]['m'] / dichtheid_fluid_backbone
    v = V_dot_leiding / opp
    return v
def toon_ontwerpparameters():
    for i in range(1, 11):
        keuze = globals().get(f"keuze_WP{i}")
        if model_WP != 'fixed':
            if keuze == "Snelheid":
               placeholder = globals().get(f"diameter_WP{i}")
               placeholder.write(f"Diameter: {warmtepompen[f"WP{i}"]['diameter']*100:.3f} cm")
            elif keuze == "Diameter":
                placeholder = globals().get(f"snelheid_WP{i}")
                placeholder.write(f"Snelheid: {warmtepompen[f"WP{i}"]['snelheid']:.3f} m/s")

    range_vals = range(1, 8) if WP7_8_9_checkbox else range(1, 7)
    for i in range_vals:
        leidingen_hot[i].write(f"Lengte: {leidingen[i]['hot']['lengte']} m")
        if keuze_hot[i] == "Snelheid":
            leidingen_d_hot[i].write(f"Diameter: {leidingen[i]['hot']['diameter']*100:.3f} cm")
        elif keuze_hot[i] == "Diameter":
            leidingen_s_hot[i].write(f"Snelheid: {leidingen[i]['hot']['snelheid']:.3f} m/s")
        if keuze_cold[i] == "Snelheid":
            leidingen_d_cold[i].write(f"Diameter: {leidingen[i]['cold']['diameter']*100:.3f} cm")
        elif keuze_cold[i] == "Diameter":
             leidingen_s_cold[i].write(f"Snelheid: {leidingen[i]['cold']['snelheid']:.3f} m/s")
def T_daling_warmtepomp(WP, T_in):
    massadebiet = warmtepompen[WP]["m"]
    T_in_source = T_in
    warmtepompen[WP]["T_in_source"] = T_in_source
    T_req_building = warmtepompen[WP]["T_req_building"]
    Q_req_building = warmtepompen[WP]["Q_req_building"]
    if Q_req_building == 0:
        T_req_building = 0

    if not WP7_8_9_checkbox:
        if WP == 'WP7' or WP == 'WP8' or WP == 'WP9':
            return 0
    warmtepompen[WP]["drukverlies"] += (massadebiet * 3600 * 1.5)
    if model_WP == "fixed":
        warmtepompen[WP]["T_building"] = T_req_building
        percentage = warmtepompen[WP]["percentage"]
        COP = COP_fixed
        warmtepompen[WP]["COP"] = COP
        Q_building = percentage * Q_req_building
        warmtepompen[WP]["Q_building"] = Q_building
        P_compressor = Q_building / COP
        warmtepompen[WP]["P_compressor"] = P_compressor
        Q_evap = Q_building - P_compressor
        T_out_source = -Q_evap / (massadebiet * Cp_fluid_backbone) + T_in_source
        warmtepompen[WP]["T_out_source"] = T_out_source
        warmtepompen[WP]["delta_T"] = T_in_source - T_out_source

        return T_in_source - T_out_source

    else:
        selected_model = warmtepompen[WP]["selected_model"]
        COP, P_compressor = bereken_variabele_COP_en_P(T_in_source, T_req_building, selected_model, WP)
        warmtepompen[WP]["COP"] = COP
        warmtepompen[WP]["P_compressor"] = P_compressor
        Q_building = COP * P_compressor
        warmtepompen[WP]["Q_building"] = Q_building
        if Q_building > Q_req_building or Q_building == Q_req_building:
            Q_building = Q_req_building
            warmtepompen[WP]['Q_building'] = Q_building
            warmtepompen[WP]["percentage"] = 1
            if COP != 0:
                P_compressor = Q_building/COP
                warmtepompen[WP]["P_compressor"] = P_compressor
        else:
            warmtepompen[WP]["percentage"] = Q_building/Q_req_building
        if Q_building == 0:
            warmtepompen_state[WP]['state'] = 'uit'
            delta_T = 0
            warmtepompen[WP]["delta_T"] = delta_T
            warmtepompen[WP]["m"] = 0
        else:
            if warmtepompen[WP]['delta_T'] == 0:
                warmtepompen[WP]['delta_T'] = warmtepompen_state[WP]['delta_T_onthoud']
                warmtepompen_state[WP]['state'] = 'aan'
            delta_T = warmtepompen[WP]["delta_T"]
            Q_evap = warmtepompen[WP]["Q_building"] - warmtepompen[WP]["P_compressor"]
            warmtepompen[WP]["m"] = Q_evap / (delta_T * Cp_fluid_backbone)
        warmtepompen[WP]["T_in_source"] = T_in
        warmtepompen[WP]["T_out_source"] = T_in - delta_T
        return delta_T
def bereken_variabele_COP_en_P(T_in, T_out, model, WP):


    #for i in range(len(model)):
    #    if model[i-1] == "_":
    #        HP_model = model[0:i-1]
    #        if model[i+2] == "l":
    #            fit = "bilinear"
    #        if model[i+2] == "q":
    #            fit = "biquadratic"

    data = HP_data[model]["data"]
    T_max = HP_data[model]["T_max"]

    if T_out > T_max:
        T_out = T_max

    if T_out == 0:
        COP = 0
        P = 0
        warmtepompen[WP]["T_building"] = T_out
        return COP,P
    warmtepompen[WP]["T_building"] = T_out

    T_in_data = data[:, 0]
    T_out_data = data[:, 1]
    COP_data = data[:, 2]
    P_data = data[:, 3]

    if fit == "bilinear":

        X = np.column_stack([
            np.ones_like(T_in_data),
            T_in_data,
            T_out_data,
            T_in_data * T_out_data
        ])

        params_COP, residuals, rank, s = lstsq(X, COP_data)
        params_P, residuals, rank, s = lstsq(X, P_data)
        A_COP, B_COP, C_COP, D_COP = params_COP
        A_P, B_P, C_P, D_P = params_P

        COP = A_COP + B_COP * T_in + C_COP * T_out + D_COP * T_in * T_out
        P = A_P + B_P * T_in + C_P * T_out + D_P * T_in * T_out

        return COP, P

    elif fit == "biquadratic":

        X = np.column_stack([
            np.ones_like(T_in_data),
            T_in_data,
            T_out_data,
            T_in_data * T_out_data,
            T_in_data**2,
            T_out_data**2,
            T_in_data**2 * T_out_data,
            T_in_data * T_out_data**2,
            T_in_data**2 * T_out_data**2
        ])

        params_COP, residuals, rank, s = lstsq(X, COP_data)
        params_P, residuals, rank, s = lstsq(X, P_data)
        A_COP, B_COP, C_COP, D_COP, E_COP, F_COP, G_COP, H_COP, I_COP = params_COP
        A_P, B_P, C_P, D_P, E_P, F_P, G_P, H_P, I_P = params_P

        COP = A_COP + B_COP*T_in + C_COP*T_out + D_COP*T_in*T_out + E_COP*T_in**2 + F_COP*T_out**2 + \
            G_COP*T_in**2*T_out + H_COP*T_in*T_out**2 + I_COP*T_in**2*T_out**2
        P = A_P + B_P*T_in + C_P*T_out + D_P*T_in*T_out + E_COP*T_in**2 + F_COP*T_out**2 + \
            G_COP*T_in**2*T_out + H_COP*T_in*T_out**2 + I_COP*T_in**2*T_out**2

        return COP, P
    else:
        sys.exit()
def T_daling_leiding(begin_Temperatuur,lengte,massadebiet,pipe_diameter,flowspeed,T_ground,x):

    drukverlies_leiding(lengte, pipe_diameter, flowspeed,massadebiet,x)

    if massadebiet == 0:
        return 0

    h_conv = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe=massadebiet,
        r_in=pipe_diameter/2,
        mu_f=mu_fluid_backbone,
        rho_f=dichtheid_fluid_backbone,
        k_f=k_fluid_backbone,
        cp_f=Cp_fluid_backbone,
        epsilon=epsilon_steel)

    R_conv = 1/(math.pi*pipe_diameter*h_conv)
    R_steel = math.log((0.5*pipe_diameter+pipe_thickness)/(0.5*pipe_diameter)) / (2*math.pi*k_steel)
    R_ground = math.log((4 * depth) / pipe_diameter) / (2 * math.pi * k_ground)
    R_tot = R_conv + R_steel + R_ground

    T_diff_tot = 0
    T_in = begin_Temperatuur
    for i in range(lengte*2):
        q = (T_in - T_ground) / R_tot
        T_diff = (q * 0.5) / (Cp_fluid_backbone * massadebiet)
        if T_diff < 0:
            T_diff = 0
        T_diff_tot = T_diff_tot + T_diff
        T_in = T_in - T_diff

    return T_diff_tot

def drukverlies_leiding(lengte,pipe_diameter,flowspeed,massadebiet,x):
    if flowspeed == 0 or massadebiet == 0:
        drukverlies = 0

    elif (x == "WP7" or x == "WP8" or x == "WP9" or x == "7") and not WP7_8_9_checkbox:
        drukverlies = 0

    else:
        f = gt.pipes.fluid_friction_factor_circular_pipe(
            m_flow_pipe=massadebiet,
            r_in=pipe_diameter / 2,
            mu_f=mu_fluid_backbone,
            rho_f=dichtheid_fluid_backbone,
            epsilon=epsilon_steel)

        drukverlies = (f * dichtheid_fluid_backbone * lengte * flowspeed ** 2) / (2 * pipe_diameter)

    if len(x) == 1:
        leidingen[float(x)]["drukverlies"] += drukverlies
    elif len(x) == 3:
        warmtepompen[x]["drukverlies"] += drukverlies

def T_daling_totaal(T_1,T_19,m_dot_backbone,T_amb):
    global T_I1
    global T_I2
    global T_I3
    global T_I4
    global T_I5
    global T_I6
    global T_I7
    global T_I8
    global T_I9
    global T_I10

    ##############################################################
    m_dot_backbone = update_massadebieten(m_dot_backbone)
    range_vals = range(1, 8) if WP7_8_9_checkbox else range(1, 7)
    for i in range_vals:
        snelheid = leidingen[i]['hot'].get('snelheid')
        diameter = leidingen[i]['hot'].get('diameter')
        if snelheid is None or diameter is None:
            if keuze_hot[i] == "Snelheid":
                snelheid = leidingen[i]['hot']['snelheid']
                diameter = bereken_diameter_pipe(i, snelheid, 'hot')
                leidingen[i]['hot']['diameter'] = diameter
            elif keuze_hot[i] == "Diameter":
                diameter = leidingen[i]['hot']['diameter']
                snelheid = bereken_snelheid_pipe(i, diameter, 'hot')
                leidingen[i]['hot']['snelheid'] = snelheid

            if keuze_cold[i] == "Snelheid":
                snelheid = leidingen[i]['cold']['snelheid']
                diameter = bereken_diameter_pipe(i, snelheid, 'cold')
                leidingen[i]['cold']['diameter'] = diameter
            elif keuze_cold[i] == "Diameter":
                diameter = leidingen[i]['cold']['diameter']
                snelheid = bereken_snelheid_pipe(i, diameter, 'cold')
                leidingen[i]['cold']['snelheid'] = snelheid
    global  eerste_keer_berekend
    if not eerste_keer_berekend:
        for i in range(1, 11):
            keuze = globals().get(f"keuze_WP{i}")
            wp_id = f'WP{i}'
            if warmtepompen[wp_id]['model'] != 'fixed':
                if keuze == "Snelheid":
                    snelheid = globals().get(f"snelheid_WP{i}")
                    if snelheid is not None:
                        diameter = bereken_diameter(wp_id, snelheid)
                        warmtepompen[wp_id]['diameter'] = diameter
                elif keuze == "Diameter":
                    diameter = globals().get(f"diameter_WP{i}")
                    if diameter is not None:
                        snelheid = bereken_snelheid(wp_id, diameter)
                        warmtepompen[wp_id]['snelheid'] = snelheid
        eerste_keer_berekend = True


    T_18 = T_1


    #st.write(L_1_2, leidingen[1]['hot']['diameter'])
    #st.write(L_1_2, leidingen[1]['hot']['snelheid'])
    # warmtepomp 10
    T_20 = T_18
    T_I10 = T_20
    T_I10_O10 = T_daling_warmtepomp("WP10", T_I10)
    T_O10 = T_I10 - T_I10_O10

    # warmtepomp 1

    T_20_2 = T_daling_leiding(T_20, L_1_2, leidingen[1]['hot']['m'], leidingen[1]['hot']['diameter'],
                              leidingen[1]['hot']['snelheid'], T_amb, "1")
    T_2 = T_20 - T_20_2
    T_2_3 = T_daling_leiding(T_2, L_2_3, leidingen[2]['hot']['m'], leidingen[2]['hot']['diameter'],
                             leidingen[2]['hot']['snelheid'], T_amb, "2")
    T_3 = T_2 - T_2_3
    T_3_I1 = T_daling_leiding(T_3, L_3_I1, warmtepompen['WP1']['m'], warmtepompen['WP1']['diameter'],
                              warmtepompen['WP1']['snelheid'], T_amb, "WP1")
    T_I1 = T_3 - T_3_I1

    T_I1_O1 = T_daling_warmtepomp("WP1", T_I1)
    T_O1 = T_I1 - T_I1_O1

    T_O1_14 = T_daling_leiding(T_O1, L_O1_14, warmtepompen['WP1']['m'], warmtepompen['WP1']['diameter'],
                               warmtepompen['WP1']['snelheid'], T_amb, "WP1")
    T_14_A = T_O1 - T_O1_14

    # warmtepomp 2

    T_3_I2 = T_daling_leiding(T_3, L_3_I2, warmtepompen['WP2']['m'], warmtepompen['WP2']['diameter'],
                              warmtepompen['WP2']['snelheid'], T_amb, "WP2")
    T_I2 = T_3 - T_3_I2

    T_I2_O2 = T_daling_warmtepomp("WP2", T_I2)
    T_O2 = T_I2 - T_I2_O2

    T_O2_14 = T_daling_leiding(T_O2, L_O2_14, warmtepompen['WP2']['m'], warmtepompen['WP2']['diameter'],
                               warmtepompen['WP2']['snelheid'], T_amb, "WP2")
    T_14_B = T_O2 - T_O2_14

    # retour

    T_14 = meng(T_14_A, warmtepompen['WP1']['m'], T_14_B, warmtepompen['WP2']['m'])

    T_14_T_15 = T_daling_leiding(T_14, L_14_15, leidingen[2]['cold']['m'], leidingen[2]['cold']['diameter'],
                                 leidingen[2]['cold']['snelheid'], T_amb, "2")
    T_15_B = T_14 - T_14_T_15

    # warmtepomp 3

    T_2_4 = T_daling_leiding(T_2, L_2_4, leidingen[3]['hot']['m'], leidingen[3]['hot']['diameter'],
                             leidingen[3]['hot']['snelheid'], T_amb, "3")
    T_4 = T_2 - T_2_4

    T_4_I3 = T_daling_leiding(T_4, L_4_I3, warmtepompen['WP3']['m'], warmtepompen['WP3']['diameter'],
                              warmtepompen['WP3']['snelheid'], T_amb, "WP3")
    T_I3 = T_4 - T_4_I3

    T_I3_O3 = T_daling_warmtepomp("WP3", T_I3)
    T_O3 = T_I3 - T_I3_O3

    T_O3_13 = T_daling_leiding(T_O3, L_O3_13, warmtepompen['WP3']['m'], warmtepompen['WP3']['diameter'],
                               warmtepompen['WP3']['snelheid'], T_amb, "WP3")
    T_13_A = T_O3 - T_O3_13

    # warmtepomp 4

    T_4_5 = T_daling_leiding(T_4, L_4_5, leidingen[4]['hot']['m'], leidingen[4]['hot']['diameter'],
                             leidingen[4]['hot']['snelheid'], T_amb, "4")
    T_5 = T_4 - T_4_5

    T_5_I4 = T_daling_leiding(T_5, L_5_I4, warmtepompen['WP4']['m'], warmtepompen['WP4']['diameter'],
                              warmtepompen['WP4']['snelheid'], T_amb, "WP4")
    T_I4 = T_5 - T_5_I4

    T_I4_O4 = T_daling_warmtepomp("WP4", T_I4)
    T_O4 = T_I4 - T_I4_O4

    T_O4_12 = T_daling_leiding(T_O4, L_O4_12, warmtepompen['WP4']['m'], warmtepompen['WP4']['diameter'],
                               warmtepompen['WP4']['snelheid'], T_amb, "WP4")
    T_12_A = T_O4 - T_O4_12

    # warmtepomp 5

    T_5_6 = T_daling_leiding(T_5, L_5_6, leidingen[5]['hot']['m'], leidingen[5]['hot']['diameter'],
                             leidingen[5]['hot']['snelheid'], T_amb, "5")
    T_6 = T_5 - T_5_6

    T_6_I5 = T_daling_leiding(T_6, L_6_I5, warmtepompen['WP5']['m'], warmtepompen['WP5']['diameter'],
                              warmtepompen['WP5']['snelheid'], T_amb, "WP5")
    T_I5 = T_6 - T_6_I5

    T_I5_O5 = T_daling_warmtepomp("WP5", T_I5)
    T_O5 = T_I5 - T_I5_O5

    T_O5_11 = T_daling_leiding(T_O5, L_O5_11, warmtepompen['WP5']['m'], warmtepompen['WP5']['diameter'],
                               warmtepompen['WP5']['snelheid'], T_amb, "WP5")
    T_11_A = T_O5 - T_O5_11

    # warmtepomp 6

    T_6_7 = T_daling_leiding(T_6, L_6_7, leidingen[6]['hot']['m'], leidingen[6]['hot']['diameter'],
                             leidingen[6]['hot']['snelheid'], T_amb, "6")
    T_7 = T_6 - T_6_7

    T_7_I6 = T_daling_leiding(T_7, L_7_I6, warmtepompen['WP6']['m'], warmtepompen['WP6']['diameter'],
                              warmtepompen['WP6']['snelheid'], T_amb, "WP6")
    T_I6 = T_7 - T_7_I6

    T_I6_O6 = T_daling_warmtepomp("WP6", T_I6)
    T_O6 = T_I6 - T_I6_O6

    T_O6_10 = T_daling_leiding(T_O6, L_O6_10, warmtepompen['WP6']['m'], warmtepompen['WP6']['diameter'],
                               warmtepompen['WP6']['snelheid'], T_amb, "WP6")
    T_10_A = T_O6 - T_O6_10

    # warmtepomp 7

    T_7_8 = T_daling_leiding(T_7, L_7_8, leidingen[7]['hot']['m'], leidingen[7]['hot']['diameter'],
                             leidingen[7]['hot']['snelheid'], T_amb, "7")
    T_8 = T_7 - T_7_8

    T_8_I7 = T_daling_leiding(T_8, L_8_I7, warmtepompen['WP7']['m'], warmtepompen['WP7']['diameter'],
                              warmtepompen['WP7']['snelheid'], T_amb, "WP7")
    T_I7 = T_8 - T_8_I7

    T_I7_O7 = T_daling_warmtepomp("WP7", T_I7)
    T_O7 = T_I7 - T_I7_O7

    # warmtepomp 8

    T_8_I8 = T_daling_leiding(T_8, L_8_I8, warmtepompen['WP8']['m'], warmtepompen['WP8']['diameter'],
                              warmtepompen['WP8']['snelheid'], T_amb, "WP8")
    T_I8 = T_8 - T_8_I8

    T_I8_O8 = T_daling_warmtepomp("WP8", T_I8)
    T_O8 = T_I8 - T_I8_O8

    # warmtepomp 9

    T_8_I9 = T_daling_leiding(T_8, L_8_I9, warmtepompen['WP9']['m'], warmtepompen['WP9']['diameter'],
                              warmtepompen['WP9']['snelheid'], T_amb, "WP9")
    T_I9 = T_8 - T_8_I9

    T_I9_O9 = T_daling_warmtepomp("WP9", T_I9)
    T_O9 = T_I9 - T_I9_O9

    # retour

    T_O7_9 = T_daling_leiding(T_O7, L_O7_9, warmtepompen['WP7']['m'], warmtepompen['WP7']['diameter'],
                              warmtepompen['WP7']['snelheid'], T_amb, "WP7")
    T_9_A = T_O7_9

    T_O8_9 = T_daling_leiding(T_O8, L_O8_9, warmtepompen['WP8']['m'], warmtepompen['WP8']['diameter'],
                              warmtepompen['WP8']['snelheid'], T_amb, "WP8")
    T_9_B = T_O8_9

    T_O9_9 = T_daling_leiding(T_O9, L_O9_9, warmtepompen['WP9']['m'], warmtepompen['WP9']['diameter'],
                              warmtepompen['WP9']['snelheid'], T_amb, "WP9")
    T_9_C = T_O9_9

    T_9_1 = meng(T_9_A, warmtepompen['WP6']['m'], T_9_B, warmtepompen['WP8']['m'])
    T_9 = meng(T_9_1, (warmtepompen['WP7']['m'] + warmtepompen['WP8']['m']), T_9_C, warmtepompen['WP9']['m'])

    T_9_10 = T_daling_leiding(T_9, L_9_10, leidingen[7]['cold']['m'], leidingen[7]['cold']['diameter'],
                              leidingen[7]['cold']['snelheid'], T_amb, "7")
    T_10_B = T_9 - T_9_10

    T_10 = meng(T_10_A, warmtepompen['WP6']['m'], T_10_B, leidingen[7]['cold']['m'])

    T_10_11 = T_daling_leiding(T_10, L_10_11, leidingen[6]['cold']['m'], leidingen[6]['cold']['diameter'],
                               leidingen[6]['cold']['snelheid'], T_amb, "6")
    T_11_B = T_10 - T_10_11

    T_11 = meng(T_11_A, warmtepompen['WP5']['m'], T_11_B, leidingen[6]['cold']['m'])

    T_11_12 = T_daling_leiding(T_11, L_11_12, leidingen[5]['cold']['m'], leidingen[5]['cold']['diameter'],
                               leidingen[5]['cold']['snelheid'], T_amb, "5")
    T_12_B = T_11 - T_11_12

    T_12 = meng(T_12_A, warmtepompen['WP4']['m'], T_12_B, leidingen[5]['cold']['m'])

    T_12_13 = T_daling_leiding(T_12, L_12_13, leidingen[4]['cold']['m'], leidingen[4]['cold']['diameter'],
                               leidingen[4]['cold']['snelheid'], T_amb, "4")
    T_13_B = T_12 - T_12_13

    T_13 = meng(T_13_A, warmtepompen['WP3']['m'], T_13_B, leidingen[4]['cold']['m'])

    T_13_15 = T_daling_leiding(T_13, L_13_15, leidingen[3]['cold']['m'], leidingen[3]['cold']['diameter'],
                               leidingen[3]['cold']['snelheid'], T_amb, "3")
    T_15_A = T_13 - T_13_15

    T_15 = meng(T_15_A, leidingen[3]['cold']['m'], T_15_B, leidingen[2]['cold']['m'])

    T_15_16 = T_daling_leiding(T_15, L_15_16, leidingen[1]['cold']['m'], leidingen[1]['cold']['diameter'],
                               leidingen[1]['cold']['snelheid'], T_amb, "1")
    T_16_A = T_15 - T_15_16

    T_16_B = T_O10

    T_16 = meng(T_16_A, leidingen[1]['cold']['m'], T_16_B, warmtepompen['WP10']['m'])

    T_19 = T_16


    ##############################################################

    solution['T2'] = T_2
    solution['T3'] = T_3
    solution['T4'] = T_4
    solution['T14'] = T_14
    solution['T15'] = T_15
    solution['T WP1 IN'] = T_I1
    solution['T WP1 OUT'] = T_O1
    solution['T WP2 IN'] = T_I2
    solution['T WP2 OUT'] = T_O2
    solution['T WP3 IN'] = T_I3
    solution['T WP3 OUT'] = T_O3
    solution['T WP4 IN'] = T_I4
    solution['T WP4 OUT'] = T_O4
    solution['T WP5 IN'] = T_I5
    solution['T WP5 OUT'] = T_O5
    solution['T WP6 IN'] = T_I6
    solution['T WP6 OUT'] = T_O6
    solution['T WP7 IN'] = T_I7
    solution['T WP7 OUT'] = T_O7
    solution['T WP8 IN'] = T_I8
    solution['T WP8 OUT'] = T_O8
    solution['T WP9 IN'] = T_I9
    solution['T WP9 OUT'] = T_O9
    solution['T WP10 IN'] = T_I10
    solution['T WP10 OUT'] = T_O10

    return T_1 - T_19
def bereken_massadebieten_in_leidingen():
    global m_1_18
    global m_18_20
    global m_20_2
    global m_2_3
    global m_WP1
    global m_WP2
    global m_2_4
    global m_WP3
    global m_4_5
    global m_WP4
    global m_5_6
    global m_WP5
    global m_6_7
    global m_WP6
    global m_7_8
    global m_WP7
    global m_WP8
    global m_WP9
    global m_9_10
    global m_10_11
    global m_11_12
    global m_12_13
    global m_13_15
    global m_14_15
    global m_15_16
    global m_16_17
    global m_17_19
    global m_WP10
    global m_retour

    m_1_18 = m_dot_backbone
    m_18_20 = m_1_18
    m_WP10 = X_WP10 * m_dot_backbone
    m_20_2 = m_18_20 - m_WP10
    m_retour = 0
    m_WP1 = X_WP1 * m_dot_backbone
    m_WP2 = X_WP2 * m_dot_backbone
    m_2_3 = m_WP1 + m_WP2
    m_14_15 = m_2_3

    m_2_4 = m_dot_backbone - m_2_3
    m_WP3 = X_WP3 * m_dot_backbone
    m_4_5 = m_2_4 - m_WP3
    m_WP4 = X_WP4 * m_dot_backbone
    m_5_6 = m_4_5 - m_WP4
    m_WP5 = X_WP5 * m_dot_backbone
    m_6_7 = m_5_6 - m_WP5
    m_WP6 = X_WP6 * m_dot_backbone
    m_WP7 = X_WP7 * m_dot_backbone
    m_WP8 = X_WP8 * m_dot_backbone
    m_WP9 = X_WP9 * m_dot_backbone
    m_7_8 = m_WP7 + m_WP8 + m_WP9
    m_9_10 = m_7_8
    m_10_11 = m_6_7
    m_11_12 = m_5_6
    m_12_13 = m_4_5
    m_13_15 = m_2_4

    m_15_16 = m_14_15 + m_13_15
    m_16_17 = m_15_16 + m_WP10
    m_17_19 = m_16_17

    if m_1_18 != m_dot_backbone:
        print("Er is een fout me de massadebieten")
def meng(T1,m1,T2,m2):

    if m1 == 0:
        return T2

    if m2 == 0:
        return T1

    T_new = (T1*m1+T2*m2)/(m1+m2)
    return T_new
def update_massadebieten(m_dot_backbone):
    m_dot_backbone = warmtepompen['WP1']['m'] + warmtepompen['WP2']['m'] + warmtepompen['WP3']['m'] + \
                     warmtepompen['WP4']['m'] + warmtepompen['WP5']['m'] + warmtepompen['WP6']['m'] + \
                     warmtepompen['WP7']['m'] + warmtepompen['WP8']['m'] + warmtepompen['WP9']['m'] + \
                     warmtepompen['WP10']['m']
    m_1_18 = m_dot_backbone
    m_18_20 = m_1_18
    m_20_2 = m_18_20 - warmtepompen['WP10']['m']
    leidingen[1]['hot']['m'] = m_20_2
    m_2_3 = warmtepompen['WP1']['m'] + warmtepompen['WP2']['m']
    leidingen[2]['hot']['m'] = m_2_3
    m_14_15 = m_2_3
    leidingen[2]['cold']['m'] = m_14_15
    m_2_4 = m_20_2 - m_2_3
    leidingen[3]['hot']['m'] = m_2_4
    m_4_5 = m_2_4 - warmtepompen['WP3']['m']
    leidingen[4]['hot']['m'] = m_4_5
    m_5_6 = m_4_5 - warmtepompen['WP4']['m']
    leidingen[5]['hot']['m'] = m_5_6
    m_6_7 = m_5_6 - warmtepompen['WP5']['m']
    leidingen[6]['hot']['m'] = m_6_7
    m_7_8 = warmtepompen['WP7']['m'] + warmtepompen['WP8']['m'] + warmtepompen['WP9']['m']
    leidingen[7]['hot']['m'] = m_7_8
    m_9_10 = m_7_8
    leidingen[7]['cold']['m'] = m_9_10
    m_10_11 = m_6_7
    leidingen[6]['cold']['m'] = m_10_11
    m_11_12 = m_5_6
    leidingen[5]['cold']['m'] = m_11_12
    m_12_13 = m_4_5
    leidingen[4]['cold']['m'] = m_12_13
    m_13_15 = m_2_4
    leidingen[3]['cold']['m'] = m_13_15
    m_15_16 = m_14_15 + m_13_15
    leidingen[1]['cold']['m'] = m_15_16
    m_16_17 = m_15_16 + warmtepompen['WP10']['m']
    m_17_19 = m_16_17
    m_dot_backbone = m_17_19
    return m_dot_backbone
def teken_schema(solution):
    leiding_dikte = 2  # Dikte van leidingen
    kader_grootte = 0.5  # Grootte temperatuurkaders
    wp_grootte = 1  # Warmtepomp-grootte
    schaal_factor = 25
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.set_aspect('equal')  # Zorgt ervoor dat cirkels geen ovalen worden
    x_min, x_max = ax.get_xlim()


    # 🏭 Leidingen
    leidingen = [
        # naar WW
        ((30, 2.9), (29, 2.9),"purple"),
        ((30, 2.9), (30, 1), "purple"),
        ((30, 3.1), (29, 3.1),"orange"),
        ((30, 3.1), (30, 6), "orange"),
        # vertrek uit WW
        ((28, 3.1), (25, 3.1),"red"),
        ((28, 2.9), (25, 2.9),"blue"),
        # naar links
        ((25, 3.1), (23, 3.1),"red"),
        ((25, 2.9), (23, 2.9),"blue"),
        # naar WP1
        ((23.1, 3.1), (23.1, 4),"red"),
        ((22.9, 2.9), (22.9, 4),"blue"),
        # naar links
        ((23, 3.1), (20.9, 3.1),"red"),
        ((23, 2.9), (21.1, 2.9),"blue"),
        # naar WP2
        ((21.1, 2.9), (21.1, 2),"blue"),
        ((20.9, 3.1), (20.9, 2),"red"),
        # naar boven
        ((25.1, 2.9), (25.1, 6), "blue"),
        ((24.9, 3.1), (24.9, 6), "red"),
        # naar WP 3
        ((25.1, 5.9), (27, 5.9), "blue"),
        ((24.9, 6.1), (27, 6.1), "red"),
        # nog naar boven
        ((25.1, 6), (25.1, 7.1), "blue"),
        ((24.9, 6), (24.9, 6.9), "red"),
        # bovenleiding naar WP6
        ((25.1, 7.1), (16.9, 7.1), "blue"),
        ((24.9, 6.9), (17.1, 6.9), "red"),
        # naar WP 4
        ((20.9, 7.1), (20.9, 6), "blue"),
        ((21.1, 6.9), (21.1, 6), "red"),
        # naar WP 5
        ((18.9, 7.1), (18.9, 5.5), "blue"),
        ((19.1, 6.9), (19.1, 5.5), "red"),
        # naar WP 6
        ((16.9, 7.1), (16.9, 6), "blue"),
        ((17.1, 6.9), (17.1, 6), "red"),
        # naar WP 10
        ((27.5, 3.1), (27.5, 3.7), "red"),
        ((27.3, 2.9), (27.3, 3.7), "blue"),

    ]
    if WP7_8_9_checkbox:
        leidingen.extend([
            # uiterst links
            ((16.9, 7.1), (14.9, 7.1), "blue"),
            ((17.1, 6.9), (15.1, 6.9), "red"),
            # naar beneden
            ((14.9, 7.1), (14.9, 4), "blue"),
            ((15.1, 6.9), (15.1, 4), "red"),
            # naar WP 9
            ((14.9, 4), (14.9, 2.5), "blue"),
            ((15.1, 4), (15.1, 2.5), "red"),
            # naar WP 8
            ((14.9, 4.1), (16.5, 4.1), "blue"),
            ((15.1, 3.9), (16.5, 3.9), "red"),
            # naar WP 7
            ((14.9, 4.1), (13.5, 4.1), "blue"),
            ((15.1, 3.9), (13.5, 3.9), "red"),
        ])

    for (start, eind, color) in leidingen:
        if color == "gradient":  # Speciale handling voor kleurverloop
            x = np.full(100, start[0])  # x blijft constant (verticale lijn)
            y = np.linspace(start[1], eind[1], 100)  # Opdelen in kleine segmenten

            # Maak een kleurverloop van rood naar blauw
            colors = [mcolors.to_rgba(c) for c in ["red", "blue"]]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

            for i in range(len(y) - 1):
                ax.plot([x[i], x[i]], [y[i], y[i + 1]], color=cmap(i / len(y)), linewidth=leiding_dikte)

        else:
            ax.plot([start[0], eind[0]], [start[1], eind[1]], color=color, linewidth=leiding_dikte)


    # ⚙️ Warmtewisselaar
    ww = patches.Circle((28.5, 3), radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(ww)
    ax.text(28.5, 3, "WW", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    imec = patches.Rectangle((29.25, 6), height=0.7, width=1.5, color="orange", ec="black")
    ax.add_patch(imec)
    ax.text(30, 6.35, "IMEC", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    dijle = patches.Rectangle((29.25, 1), height=0.7, width=1.5, color="purple", ec="black")
    ax.add_patch(dijle)
    ax.text(30, 1.35, "DIJLE", ha="center", va="center", fontsize=wp_grootte * schaal_factor, fontweight="bold")

    # ⚙️ Warmtepomp
    pos_WP1 = (23,4.5)
    pos_WP2 = (21,1.5)
    pos_WP3 = (27.5,6)
    pos_WP4 = (21,5.5)
    pos_WP5 = (19,5)
    pos_WP6 = (17,5.5)
    if WP7_8_9_checkbox:
        pos_WP7 = (13,4)
        pos_WP8 = (17,4)
        pos_WP9 = (15,2)
    pos_WP10 = (27.4,4.2)



    WP1 = patches.Circle(pos_WP1, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP1)
    ax.text(pos_WP1[0],pos_WP1[1], "WP1", ha="center", va="center", fontsize=wp_grootte*schaal_factor, fontweight="bold")

    WP2 = patches.Circle(pos_WP2, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP2)
    ax.text(pos_WP2[0], pos_WP2[1], "WP2", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP3 = patches.Circle(pos_WP3, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP3)
    ax.text(pos_WP3[0], pos_WP3[1], "WP3", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP4 = patches.Circle(pos_WP4, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP4)
    ax.text(pos_WP4[0], pos_WP4[1], "WP4", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP5 = patches.Circle(pos_WP5, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP5)
    ax.text(pos_WP5[0], pos_WP5[1], "WP5", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP6 = patches.Circle(pos_WP6, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP6)
    ax.text(pos_WP6[0], pos_WP6[1], "WP6", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
            fontweight="bold")
    WP10 = patches.Circle(pos_WP10, radius=wp_grootte / 2, color="green", ec="black")
    ax.add_patch(WP10)
    ax.text(pos_WP10[0], pos_WP10[1], "WP10", ha="center", va="center", fontsize=wp_grootte * schaal_factor*0.85,
            fontweight="bold")
    if WP7_8_9_checkbox:
        WP7 = patches.Circle(pos_WP7, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP7)
        ax.text(pos_WP7[0], pos_WP7[1], "WP7", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
        WP8 = patches.Circle(pos_WP8, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP8)
        ax.text(pos_WP8[0], pos_WP8[1], "WP8", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
        WP9 = patches.Circle(pos_WP9, radius=wp_grootte / 2, color="green", ec="black")
        ax.add_patch(WP9)
        ax.text(pos_WP9[0], pos_WP9[1], "WP9", ha="center", va="center", fontsize=wp_grootte * schaal_factor,
                fontweight="bold")
    # 🔲 Temperatuurkaders
    temperaturen = {
        #WW
        (29.4, 3.3, "orange"): str(round(T_imec,2)) + "°C",
        (29.4, 2.7, "purple"): str(solution["T naar Dijle"]) + "°C",
        (28, 3.6, "red"): str(solution["T WARMTEWISSELAAR OUT"]) + "°C",
        (27.6, 2.7, "blue"): str(solution["T WARMTEWISSELAAR IN"]) + "°C",
        # WP1
        (23.65, 3.9, "red"): str(warmtepompen['WP1']['T_in_source']) + "°C",
        (22.35, 3.9, "blue"): str(warmtepompen['WP1']['T_out_source']) + "°C",
        # WP2
        (20.35, 2.1, "red"): str(warmtepompen['WP2']['T_in_source']) + "°C",
        (21.65, 2.1, "blue"): str(warmtepompen['WP2']['T_out_source']) + "°C",
        # WP3
        (26.6, 6.4, "red"): str(warmtepompen['WP3']['T_in_source']) + "°C",
        (26.6, 5.6, "blue"): str(warmtepompen['WP3']['T_out_source']) + "°C",
        # WP4
        (21.65, 6.1, "red"): str(warmtepompen['WP4']['T_in_source']) + "°C",
        (20.35, 6.1, "blue"): str(warmtepompen['WP4']['T_out_source']) + "°C",
        # WP5
        (19.65, 5.6, "red"): str(warmtepompen['WP5']['T_in_source']) + "°C",
        (18.35, 5.6, "blue"): str(warmtepompen['WP5']['T_out_source']) + "°C",
        # WP6
        (17.65, 6.1, "red"): str(warmtepompen['WP6']['T_in_source']) + "°C",
        (16.35, 6.1, "blue"): str(warmtepompen['WP6']['T_out_source']) + "°C",
        # WP10
        #(25.8, 3.6, "red"): str(solution["T WP10 IN"]) + "°C",
        (26.8, 3.6, "blue"): str(warmtepompen['WP10']['T_out_source']) + "°C",
    }
    if WP7_8_9_checkbox:
        temperaturen.update({
        # WP7
        (14.0, 3.7, "red"): str(warmtepompen['WP7']['T_in_source']) + "°C",
        (14.0, 4.3, "blue"): str(warmtepompen['WP7']['T_out_source']) + "°C",
        # WP8
        (16, 3.7, "red"): str(warmtepompen['WP8']['T_in_source']) + "°C",
        (16, 4.3, "blue"): str(warmtepompen['WP8']['T_out_source']) + "°C",
        # WP9
        (15.7, 2.6, "red"): str(warmtepompen['WP9']['T_in_source']) + "°C",
        (14.35, 2.6, "blue"): str(warmtepompen['WP9']['T_out_source']) + "°C",
        })
    if temp_checkbox:
        for (x, y, letter_color), temp in temperaturen.items():
            ax.text(x, y, temp, ha="center", va="center", fontsize=kader_grootte*schaal_factor+3, fontweight="bold",color=letter_color)

    leidingen_nummers = {
        # WW
        (26, 3, "black"): 1,
        (24, 3, "black"): 2,
        (25, 4.5, "black"): 3,
        (23.5, 7, "black"): 4,
        (20, 7, "black"): 5,
        (18, 7, "black"): 6,
    }
    if WP7_8_9_checkbox:
        leidingen_nummers.update({
        (15, 6, "black"): 7,
        })
    if nummer_checkbox:
        factor_nummer = kader_grootte * schaal_factor + 25
        for (x, y, letter_color), leiding in leidingen_nummers.items():
            ax.text(x, y, leiding, ha="center", va="center", fontsize=factor_nummer, fontweight="bold",
                    color=letter_color)

    # 🔧 Lay-out instellingen

    ax.set_adjustable("datalim")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig

def update_simulatie_dict(source, target):
    for wp_id, parameters in source.items():
        if wp_id not in target:
            target[wp_id] = {}
        for param, waarde in parameters.items():
            if param not in target[wp_id]:
                target[wp_id][param] = []
            target[wp_id][param].append(waarde)

##################################################
###### INPUT PARAMETERS/AANNAMES #################
##################################################

modellen = ["VM_Vitocal_350G_Pro_C075",
            "VM_Vitocal_350G_Pro_C210",
            "VM_Vitocal_300G_Pro_DS090",
            "VM_Vitocal_300G_Pro_DS230",
            "VM_Vitocal_200G_Pro_A080",
            "VM_Vitocal_200G_Pro_A100",
            "NIBE_F1355_28",
            "NIBE_F1355_43",
            "NIBE_F1345_60"]

fit = "bilinear"

snelheid_min = 0.01
snelheid_max = 3.00
snelheid_value = 2.00
snelheid_step = 0.01

diameter_min = 0.1
diameter_max = 30.0
diameter_value = 10.0
diameter_value_backbone = 10.0
diameter_step = 0.1
tekst_keuze = 'Wat wil je vastleggen?'

leidingen_hot = {}
leidingen_cold = {}
leidingen_d_hot = {}
leidingen_s_hot = {}
leidingen_d_cold = {}
leidingen_s_cold = {}
keuze_hot = {}
keuze_cold = {}


### VISUALISATIE INPUT
with st.expander("Gegevens invoeren"):
    col1, col2 = st.columns(2)

    with col1:
        T_imec = st.number_input("Temperatuur IMEC", value=21.8, step=0.1, key='T_imec')
    #with col2:
        #debiet_backbone = st.slider("Volumedebiet backbone [m^3/h]", 20, 200, value=70,key='debiet_imec')


    col3, col4 = st.columns(2)
    with col3:
        model_WP = st.selectbox("COP-model", ["variabel","vaste waarde (niet meer ondersteund sinds 1/5)"], key='model')
        #WP7_8_9_checkbox = st.checkbox("Warmtepomp 7-8-9", value=True)
    with col4:
        if model_WP == "vaste waarde (niet meer ondersteund sinds 1/5)":
            model_WP = 'fixed'
            COP_fixed = st.number_input("Vaste waarde COP:", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    col1A,col1B,col1C,col1D,col1E = st.columns(5)
    with col1A:
        WP7_8_9_checkbox = st.checkbox("Warmtepomp 7-8-9", value=True)
    with col1B:
        nummer_checkbox = st.checkbox("Toon leiding nummers", value=True)
    with col1C:
        temp_checkbox = st.checkbox("Toon temperaturen", value=True)
    if model_WP == "variabel":
        tab1, tab2, tab3, tab4 = st.tabs(["Warmtepompen 1-5", "Warmtepompen 6-10","Leidingen 1-5","Leidingen 6-7"])
        with tab1:
            col5, col6, col7, col8, col9 = st.columns(5)

            with col5:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 1</h4>", unsafe_allow_html=True)
                selected_model_WP1 = st.selectbox("Model", modellen, key='modelWP1')
                delta_T_WP1 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP1')
                keuze_WP1 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP1')
                if keuze_WP1 == "Snelheid":
                    snelheid_WP1 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP1')
                    diameter_WP1 = st.empty()
                elif keuze_WP1 == "Diameter":
                    diameter_WP1 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP1')/100
                    snelheid_WP1 = st.empty()

            with col6:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 2</h4>", unsafe_allow_html=True)
                selected_model_WP2 = st.selectbox("Model", modellen, key='modelWP2')
                delta_T_WP2 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP2')
                keuze_WP2 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP2')
                if keuze_WP2 == "Snelheid":
                    snelheid_WP2 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP2')
                    diameter_WP2 = st.empty()
                elif keuze_WP2 == "Diameter":
                    diameter_WP2 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP2')/100
                    snelheid_WP2 = st.empty()

            with col7:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 3</h4>", unsafe_allow_html=True)
                selected_model_WP3 = st.selectbox("Model", modellen, key='modelWP3')
                delta_T_WP3 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP3')
                keuze_WP3 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP3')
                if keuze_WP3 == "Snelheid":
                    snelheid_WP3 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP3')
                    diameter_WP3 = st.empty()
                elif keuze_WP3 == "Diameter":
                    diameter_WP3 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP3')/100
                    snelheid_WP3 = st.empty()

            with col8:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 4</h4>", unsafe_allow_html=True)
                selected_model_WP4 = st.selectbox("Model", modellen, key='modelWP4')
                delta_T_WP4 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP4')
                keuze_WP4 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP4')
                if keuze_WP4 == "Snelheid":
                    snelheid_WP4 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP4')
                    diameter_WP4 = st.empty()
                elif keuze_WP4 == "Diameter":
                    diameter_WP4 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP4')/100
                    snelheid_WP4 = st.empty()

            with col9:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 5</h4>", unsafe_allow_html=True)
                selected_model_WP5 = st.selectbox("Model", modellen, key='modelWP5')
                delta_T_WP5 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP5')
                keuze_WP5 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP5')
                if keuze_WP5 == "Snelheid":
                    snelheid_WP5 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP5')
                    diameter_WP5 = st.empty()
                elif keuze_WP5 == "Diameter":
                    diameter_WP5 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP5')/100
                    snelheid_WP5 = st.empty()
        with tab2:
            col10, col11, col12, col13, col14 = st.columns(5)

            with col10:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 6</h4>", unsafe_allow_html=True)
                selected_model_WP6 = st.selectbox("Model", modellen, key='modelWP6')
                delta_T_WP6 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP6')
                keuze_WP6 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP6')
                if keuze_WP6 == "Snelheid":
                    snelheid_WP6 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key='snelheidWP6')
                    diameter_WP6 = st.empty()
                elif keuze_WP6 == "Diameter":
                    diameter_WP6 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key='diameterWP6')/100
                    snelheid_WP6 = st.empty()

            with col11:
                if WP7_8_9_checkbox:
                    st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 7</h4>", unsafe_allow_html=True)
                    selected_model_WP7 = st.selectbox("Model", modellen, key='modelWP7')
                    delta_T_WP7 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP7')
                    keuze_WP7 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP7')
                    if keuze_WP7 == "Snelheid":
                        snelheid_WP7 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                                 snelheid_step, key='snelheidWP7')
                        diameter_WP7 = st.empty()
                    elif keuze_WP7 == "Diameter":
                        diameter_WP7 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                                 key='diameterWP7')/100
                        snelheid_WP7 = st.empty()
                else:
                    selected_model_WP7 = 'Not identified'
                    delta_T_WP7 = 0

            with col12:
                if WP7_8_9_checkbox:
                    st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 8</h4>", unsafe_allow_html=True)
                    selected_model_WP8 = st.selectbox("Model", modellen, key='modelWP8')
                    delta_T_WP8 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP8')
                    keuze_WP8 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP8')
                    if keuze_WP8 == "Snelheid":
                        snelheid_WP8 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                                 snelheid_step, key='snelheidWP8')
                        diameter_WP8 = st.empty()
                    elif keuze_WP8 == "Diameter":
                        diameter_WP8 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                                 key='diameterWP8')/100
                        snelheid_WP8 = st.empty()
                else:
                    selected_model_WP8 = 'Not identified'
                    delta_T_WP8 = 0

            with col13:
                if WP7_8_9_checkbox:
                    st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 9</h4>", unsafe_allow_html=True)
                    selected_model_WP9 = st.selectbox("Model", modellen, key='modelWP9')
                    delta_T_WP9 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP9')
                    keuze_WP9 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP9')
                    if keuze_WP9 == "Snelheid":
                        snelheid_WP9 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                                 snelheid_step, key='snelheidWP9')
                        diameter_WP9 = st.empty()
                    elif keuze_WP9 == "Diameter":
                        diameter_WP9 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                                 key='diameterWP9')/100
                        snelheid_WP9 = st.empty()
                else:
                    selected_model_WP9 = 'Not identified'
                    delta_T_WP9 = 0

            with col14:
                st.markdown("<h4 style='text-align: center;font-size:22px;'>WARMTEPOMP 10</h4>", unsafe_allow_html=True)
                selected_model_WP10 = st.selectbox("Model", modellen, key='modelWP10')
                delta_T_WP10 = st.slider("\u0394T bron", 2.0, 8.0, value=3.5, step=0.1, key='deltaTWP10')
                keuze_WP10 = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key='keuzeWP10')
                if keuze_WP10 == "Snelheid":
                    snelheid_WP10 = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                              snelheid_step, key='snelheidWP10')
                    diameter_WP10 = st.empty()
                elif keuze_WP10 == "Diameter":
                    diameter_WP10 = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                              key='diameterWP10')/100
                    snelheid_WP10 = st.empty()
        with tab3:
            cols = st.columns(5)

            for i, col in enumerate(cols, start=1):
                with col:
                    st.markdown(f"<h4 style='text-align: center;font-size:22px;'>LEIDING {i}</h4>",
                                unsafe_allow_html=True)

                    leidingen_hot[i] = st.empty()
                    leidingen_cold[i] = st.empty()

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>WARM</h4>", unsafe_allow_html=True)
                    keuze_hot[i] = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key=f'keuzehotL{i}')
                    if keuze_hot[i] == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidhotL{i}')
                        leidingen_d_hot[i] = st.empty()
                        leidingen[i]['hot'] = {}
                        leidingen[i]['hot']['snelheid'] = snelheid
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value_backbone, diameter_step,
                                             key=f'diameterhotL{i}') / 100
                        leidingen_s_hot[i] = st.empty()
                        leidingen[i]['hot'] = {}
                        leidingen[i]['hot']['diameter'] = diameter

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>KOUD</h4>", unsafe_allow_html=True)
                    keuze_cold[i] = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key=f'keuzecoldL{i}')
                    if keuze_cold[i] == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidcoldL{i}')
                        leidingen_d_cold[i] = st.empty()
                        leidingen[i]['cold'] = {}
                        leidingen[i]['cold']['snelheid'] = snelheid
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value_backbone, diameter_step,
                                             key=f'diametercoldL{i}') / 100
                        leidingen_s_cold[i] = st.empty()
                        leidingen[i]['cold'] = {}
                        leidingen[i]['cold']['diameter'] = diameter
        with tab4:
            cols = st.columns(5)
            for i in [6, 7] if WP7_8_9_checkbox else [6]:

                col_index = i - 6  # omdat cols[0] bij L6 hoort, en cols[1] bij L7
                with cols[col_index]:
                    st.markdown(f"<h4 style='text-align: center;font-size:22px;'>LEIDING {i}</h4>",
                                unsafe_allow_html=True)

                    leidingen_hot[i] = st.empty()
                    leidingen_cold[i] = st.empty()

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>WARM</h4>", unsafe_allow_html=True)
                    keuze_hot[i] = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key=f'keuzehotL{i}')
                    if keuze_hot[i] == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidhotL{i}')
                        leidingen_d_hot[i] = st.empty()
                        leidingen[i]['hot'] = {}
                        leidingen[i]['hot']['snelheid'] = snelheid
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value_backbone, diameter_step,
                                             key=f'diameterhotL{i}') / 100
                        leidingen_s_hot[i] = st.empty()
                        leidingen[i]['hot'] = {}
                        leidingen[i]['hot']['diameter'] = diameter

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>KOUD</h4>", unsafe_allow_html=True)
                    keuze_cold[i] = st.radio(tekst_keuze, ("Diameter", "Snelheid"), key=f'keuzecoldL{i}')
                    if keuze_cold[i] == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidcoldL{i}')
                        leidingen_d_cold[i] = st.empty()
                        leidingen[i]['cold'] = {}
                        leidingen[i]['cold']['snelheid'] = snelheid
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value_backbone, diameter_step,
                                             key=f'diametercoldL{i}') / 100
                        leidingen_s_cold[i] = st.empty()
                        leidingen[i]['cold'] = {}
                        leidingen[i]['cold']['diameter'] = diameter
            if not WP7_8_9_checkbox:
                leidingen[7]['hot'] = {}
                leidingen[7]['hot']['snelheid'] = 0
                leidingen[7]['hot']['diameter'] = 0
                leidingen[7]['cold'] = {}
                leidingen[7]['cold']['snelheid'] = 0
                leidingen[7]['cold']['diameter'] = 0

    else:
        tab1, tab2 = st.tabs(["Leidingen 1-5",'Leidingen 6-7'])
        with tab1:
            cols = st.columns(5)

            for i, col in enumerate(cols, start=1):
                with col:
                    leidingen[i] = {}
                    st.markdown(f"<h4 style='text-align: center;font-size:22px;'>LEIDING {i}</h4>",
                                unsafe_allow_html=True)

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>WARM</h4>", unsafe_allow_html=True)
                    keuze_hot = st.radio(tekst_keuze, ("Snelheid", "Diameter"), key=f'keuzehotL{i}')
                    if keuze_hot == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidhotL{i}')
                        diameter = st.empty()
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key=f'diameterhotL{i}') / 100
                        snelheid = st.empty()
                    leidingen[i]['hot'] = {'snelheid': snelheid, 'diameter': diameter}

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>KOUD</h4>", unsafe_allow_html=True)
                    keuze_cold = st.radio(tekst_keuze, ("Snelheid", "Diameter"), key=f'keuzecoldL{i}')
                    if keuze_cold == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidcoldL{i}')
                        diameter = st.empty()
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key=f'diametercoldL{i}') / 100
                        snelheid = st.empty()
                    leidingen[i]['cold'] = {'snelheid': snelheid, 'diameter': diameter}
        with tab2:
            cols = st.columns(5)
            for i in [6, 7]:
                col_index = i - 6  # omdat cols[0] bij L6 hoort, en cols[1] bij L7
                with cols[col_index]:
                    leidingen[i] = {}
                    st.markdown(f"<h4 style='text-align: center;font-size:22px;'>LEIDING {i}</h4>",
                                unsafe_allow_html=True)

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>WARM</h4>", unsafe_allow_html=True)
                    keuze_hot = st.radio(tekst_keuze, ("Snelheid", "Diameter"), key=f'keuzehotL{i}')
                    if keuze_hot == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidhotL{i}')
                        diameter = st.empty()
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key=f'diameterhotL{i}') / 100
                        snelheid = st.empty()
                    leidingen[i]['hot'] = {'snelheid': snelheid, 'diameter': diameter}

                    st.markdown("<h4 style='text-align: center;font-size:18px;'>KOUD</h4>", unsafe_allow_html=True)
                    keuze_cold = st.radio(tekst_keuze, ("Snelheid", "Diameter"), key=f'keuzecoldL{i}')
                    if keuze_cold == "Snelheid":
                        snelheid = st.slider("Snelheid (m/s)", snelheid_min, snelheid_max, snelheid_value,
                                             snelheid_step, key=f'snelheidcoldL{i}')
                        diameter = st.empty()
                    else:
                        diameter = st.slider("Diameter (cm)", diameter_min, diameter_max, diameter_value, diameter_step,
                                             key=f'diametercoldL{i}') / 100
                        snelheid = st.empty()
                    leidingen[i]['cold'] = {'snelheid': snelheid, 'diameter': diameter}





        selected_model_WP1 = 'Not identified'
        selected_model_WP2 = 'Not identified'
        selected_model_WP3 = 'Not identified'
        selected_model_WP4 = 'Not identified'
        selected_model_WP5 = 'Not identified'
        selected_model_WP6 = 'Not identified'
        selected_model_WP7 = 'Not identified'
        selected_model_WP8 = 'Not identified'
        selected_model_WP9 = 'Not identified'
        selected_model_WP10 = 'Not identified'

        delta_T_WP1 = 0
        delta_T_WP2 = 0
        delta_T_WP3 = 0
        delta_T_WP4 = 0
        delta_T_WP5 = 0
        delta_T_WP6 = 0
        delta_T_WP7 = 0
        delta_T_WP8 = 0
        delta_T_WP9 = 0
        delta_T_WP10 = 0

wp_ids = [f'WP{i}' for i in range(1, 11)]
for wp_id in wp_ids:
    warmtepompen[wp_id]['model'] = globals()[f'selected_model_{wp_id}']
    warmtepompen[wp_id]['selected_model'] = globals()[f'selected_model_{wp_id}']
    warmtepompen[wp_id]['delta_T'] = globals()[f'delta_T_{wp_id}']

keuze_simulatie = st.radio('Wat wil je berekenen?', ("Simulatie van een heel jaar","1 steady state oplossing"))
if keuze_simulatie == "Simulatie van een heel jaar":
    keuze_simulatie = 'Simulatie'
else:
    keuze_simulatie = 'SS'
###############################################################################
################### VASTE DATA ################################################
###############################################################################


### WARMTEWISSELAAR
type_WW = 'tegenstroom'
A = 350  # m²
U = 6000  # W/m²·K

### DATA WARMTEPOMPEN

## VIESSMANN

# Vitocal 350-G Pro

HP_data["VM_Vitocal_350G_Pro_C075"]["data"] = np.array([
            [0, 35, 3.74, 20500],
            [10, 35, 4.44, 21100],
            [5, 40, 3.85, 22900],
            [10, 45, 3.88, 23900],
            [15, 50, 3.88, 27000],
            [0, 55, 2.72, 22900],
            [10, 65, 2.72, 29600],
            [20, 75, 2.62, 30500]])
HP_data["VM_Vitocal_350G_Pro_C075"]["T_max"] = 45

HP_data["VM_Vitocal_350G_Pro_C210"]["data"] = np.array([
            [0, 35, 3.50, 55000],
            [10, 35, 4.59, 59000],
            [5, 40, 3.48, 63000],
            [10, 45, 3.55, 70600],
            [15, 50, 3.66, 72100],
            [0, 55, 2.64, 60000],
            [10, 65, 2.68, 79000],
            [20, 75, 2.71, 79700]])
HP_data["VM_Vitocal_350G_Pro_C210"]["T_max"] = 45

# Vitocal 300-G Pro

HP_data["VM_Vitocal_300G_Pro_DS090"]["data"] = np.array([
            [0, 25, 5.62, 15450],
            [10, 25, 6.98, 16050],
            [0, 35, 4.55, 18650],
            [10, 35, 5.66, 19210],
            [0, 45, 3.65, 22450],
            [10, 45, 4.48, 22950],
            [0, 55, 2.95, 26930],
            [5, 60, 2.89, 30330]])
HP_data["VM_Vitocal_300G_Pro_DS090"]["T_max"] = 45

HP_data["VM_Vitocal_300G_Pro_DS230"]["data"] = np.array([
            [0, 25, 5.69, 40300],
            [10, 25, 7.16, 42900],
            [0, 35, 4.60, 48300],
            [10, 35, 5.82, 50900],
            [0, 45, 3.66, 58290],
            [10, 45, 4.63, 60690],
            [0, 55, 2.98, 70520],
            [5, 60, 2.95, 79120]])
HP_data["VM_Vitocal_300G_Pro_DS230"]["T_max"] = 45

# Vitocal 200-G Pro

HP_data["VM_Vitocal_200G_Pro_A080"]["data"] = np.array([
            [0, 25, 5.88, 13290],
            [10, 25, 7.97, 12680],
            [0, 35, 4.55, 16590],
            [10, 35, 5.86, 16490],
            [0, 45, 3.56, 20320],
            [10, 45, 4.48, 20400],
            [0, 55, 2.91, 23900],
            [5, 60, 2.88, 26500]])
HP_data["VM_Vitocal_200G_Pro_A080"]["T_max"] = 45

HP_data["VM_Vitocal_200G_Pro_A100"]["data"] = np.array([
            [0, 25, 5.79, 18130],
            [10, 25, 7.79, 17380],
            [0, 35, 4.53, 22280],
            [10, 35, 5.79, 22280],
            [0, 45, 3.55, 27230],
            [10, 45, 4.45, 27380],
            [0, 55, 2.89, 32200],
            [5, 60, 2.85, 35620]])
HP_data["VM_Vitocal_200G_Pro_A100"]["T_max"] = 45

## NIBE

# F1355

HP_data["NIBE_F1355_28"]["data"] = np.array([
            [0, 35, 4.55, 4560],
            [10, 35, 5.60, 4760],
            [0, 45, 3.59, 5540],
            [10, 45, 4.40, 5840]])
HP_data["NIBE_F1355_28"]["T_max"] = 45

HP_data["NIBE_F1355_43"]["data"] = np.array([
            [0, 35, 4.38, 7100],
            [10, 35, 5.52, 7330],
            [0, 45, 3.46, 8400],
            [10, 45, 4.31, 8920]])
HP_data["NIBE_F1355_43"]["T_max"] = 45

# F1345

HP_data["NIBE_F1345_60"]["data"] = np.array([
            [0, 35, 4.32, 13720],
            [10, 35, 5.19, 15080],
            [0, 45, 3.50, 16020],
            [10, 45, 4.22, 17600]])
HP_data["NIBE_F1345_60"]["T_max"] = 45



### FLUIDS
debiet_imec = 60 #m3/h
dichtheid_fluid_imec = 997 #kg/m3
dichtheid_fluid_backbone = 997 #kg/m3
Cp_fluid_imec = 4180  # J/kg·K
Cp_fluid_backbone = 4180  # J/kg·K
k_fluid_backbone = 0.598  # W/(m*k)
mu_fluid_backbone = 0.001  # N*s/m2

debiet_backbone = 200
m_dot_imec = debiet_imec * dichtheid_fluid_imec / 3600  # kg/s
m_dot_backbone = debiet_backbone * dichtheid_fluid_backbone / 3600  # kg/s

### LEIDINGEN ONTWERPDATA
depth = 1  # m
k_ground = 1  # W/(m*K)
pipe_thickness = 0.01  # m
k_steel = 45  # W/(m*K)
epsilon_steel = 0.00005   # m
flowspeed_backbone = 2  # m/s

diff_ground = 0.005
k_LB = 1
LAI = 2
period = 31536000

### LEIDINGEN LENGTES
lengte_WP10 = 0
L_1_2 = 115  # m
L_2_3 = 30  # m
L_3_I1 = 20  # m
L_O1_14 = L_3_I1  # m
lengte_WP1 = L_3_I1
L_3_I2 = 75  # m
L_O2_14 = L_3_I2  # m
lengte_WP2 = L_3_I2
L_14_15 = L_2_3  # m
L_15_16 = L_1_2  # m

L_2_4 = 60  # m
L_4_I3 = 115  # m
L_O3_13 = L_4_I3  # m
lengte_WP3 = L_4_I3
L_13_15 = L_2_4  # m
L_4_5 = 190  # m
L_5_I4 = 40  # m
lengte_WP4 = L_5_I4
L_O4_12 = L_5_I4  # m
L_5_6 = 50  # m
L_6_I5 = 25  # m
lengte_WP5 = L_6_I5
L_O5_11 = L_6_I5  # m
L_6_7 = 100  # m
L_7_I6 = 25  # m
lengte_WP6 = L_7_I6
L_O6_10 = L_7_I6  # m
L_10_11 = L_6_7  # m
L_11_12 = L_5_6  # m
L_12_13 = L_4_5  # m
L_7_8 = 375  # m
L_8_I7 = 325  # m
lengte_WP7 = L_8_I7
L_8_I8 = 50  # m
lengte_WP8 = L_8_I8
L_8_I9 = 50  # m
lengte_WP9 = L_8_I9
L_O9_9 = L_8_I9  # m
L_O8_9 = L_8_I8  # m
L_O7_9 = L_8_I7  # m
L_9_10 = L_7_8  # m

leiding_lengtes = {
    1: L_1_2,
    2: L_2_3,
    3: L_2_4,
    4: L_4_5,
    5: L_5_6,
    6: L_6_7,
    7: L_7_8
}
range_vals = range(1,8) if WP7_8_9_checkbox else range(1, 7)
for i in range_vals:
    lengte = leiding_lengtes[i]
    leidingen[i]['hot']['lengte'] = lengte
    leidingen[i]['cold']['lengte'] = lengte


##########
percentage_WP1 = 0.70
percentage_WP2 = 0.70
percentage_WP3 = 0.70
percentage_WP4 = 0.70
percentage_WP5 = 0.70
percentage_WP6 = 0.70
percentage_WP7 = 0.70
percentage_WP8 = 0.70
percentage_WP9 = 0.70
percentage_WP10 = 0.70


###################################
##### warmtepompen dictionary #####
###################################
#warmtepompen["WP1"]["COP_Viessmann_biquadratic"] = 'not defined'
#warmtepompen["WP1"]["COP_Viessmann_bilineair"] = 'not defined'

bestand = '000-00 Verbruiksgegevens site arenberg.xlsx'
df = pd.read_excel(bestand, sheet_name='Blad1')

data = 'buiten_temperaturen.xlsx'
df_2 = pd.read_excel(data)
temp_data = df_2.iloc[:, 13].tolist()
filtered = temp_data[::3]
temperatuur = [round(x/10,1) for x in filtered]

dagelijkse_temperatuur = []
for i in range(365):
    dagelijkse_temperatuur.append(sum(temperatuur[i*24:i*24+24])/24)

grond_temperatuur = []
T_prev = 10
for i in range(365):
    grond_temperatuur.append(T_prev + (dagelijkse_temperatuur[i] - T_prev) * math.exp(-depth * math.sqrt(math.pi / (diff_ground * period))) * math.exp(-k_LB * LAI))
    T_prev = grond_temperatuur[i]



if len(temperatuur) != len(df.iloc[:, 6].tolist()):
    st.write('lengtes van lijsten zijn niet hetzelfde: kijk dit na')

if keuze_simulatie == 'SS':
    T_req_building_WP1 = [50]  # °C
    Q_req_building_WP1 = [0]  # W

    T_req_building_WP2 = [50]  # °C
    Q_req_building_WP2 = [200000]  # W

    T_req_building_WP3 = [50]  # °C
    Q_req_building_WP3 = [200000]  # W

    T_req_building_WP4 = [50]  # °C
    Q_req_building_WP4 = [200000]  # W

    T_req_building_WP5 = [50]  # °C
    Q_req_building_WP5 = [200000]  # W

    T_req_building_WP6 = [50]  # °C
    Q_req_building_WP6 = [200000]  # W

    T_req_building_WP7 = [50]  # °C
    Q_req_building_WP7 = [200000]  # W

    T_req_building_WP8 = [50]  # °C
    Q_req_building_WP8 = [200000]  # W

    T_req_building_WP9 = [50]  # °C
    Q_req_building_WP9 = [200000]  # W

    T_req_building_WP10 = [50]  # °C
    Q_req_building_WP10 = 200000  # W
else:
    T_req_building_WP1 = [-0.8333 * T + 51.667 for T in temperatuur]
    Q_req_building_WP1 = df.iloc[:, 6].tolist()
    Q_req_building_WP1 = [x * 1000 for x in Q_req_building_WP1]

    T_req_building_WP2 = [-1.2 * T + 53 for T in temperatuur]
    Q_req_building_WP2 = df.iloc[:, 7].tolist()
    Q_req_building_WP2 = [x * 1000 for x in Q_req_building_WP2]

    T_req_building_WP3 = [-0.0067 * T * T - 1.3667 * T + 57 for T in temperatuur]
    Q_req_building_WP3 = df.iloc[:, 1].tolist()
    Q_req_building_WP3 = [x * 1000 for x in Q_req_building_WP3]

    T_req_building_WP4 = [-1.32 * T + 51.8 for T in temperatuur]
    Q_req_building_WP4 = df.iloc[:, 9].tolist()
    Q_req_building_WP4 = [x * 1000 for x in Q_req_building_WP4]

    T_req_building_WP5 = [50 for T in temperatuur]
    Q_req_building_WP5 = df.iloc[:, 3].tolist()
    Q_req_building_WP5 = [x * 1000 for x in Q_req_building_WP5]

    T_req_building_WP6 = [-1.2 * T + 58 for T in temperatuur]
    Q_req_building_WP6 = df.iloc[:, 5].tolist()
    Q_req_building_WP6 = [x * 1000 for x in Q_req_building_WP6]

    T_req_building_WP7 = [0.0667 * T * T - 1.2333 * T + 42 for T in temperatuur]
    Q_req_building_WP7 = df.iloc[:, 8].tolist()
    Q_req_building_WP7 = [x * 1000 for x in Q_req_building_WP7]

    T_req_building_WP8 = [-0.0533 * T * T - 0.5333 * T + 55 for T in temperatuur]
    Q_req_building_WP8 = df.iloc[:, 4].tolist()
    Q_req_building_WP8 = [x * 1000 for x in Q_req_building_WP8]

    T_req_building_WP9 = [0.0668 * T * T - 2.0017 * T + 48.3 for T in temperatuur]
    Q_req_building_WP9 = df.iloc[:, 2].tolist()
    Q_req_building_WP9 = [x * 1000 for x in Q_req_building_WP9]

    T_req_building_WP10 = [-0.4 * T + 51 for T in temperatuur]
    Q_req_building_WP10 = df.iloc[:, 0].tolist()
    Q_req_building_WP10 = [x * 1000 for x in Q_req_building_WP10]

### WARMTEPOMPEN DEBIETEN

if WP7_8_9_checkbox:
    X_WP1 = 0.1
    X_WP2 = 0.1
    X_WP3 = 0.1
    X_WP4 = 0.1
    X_WP5 = 0.1
    X_WP6 = 0.1
    X_WP7 = 0.1
    X_WP8 = 0.1
    X_WP9 = 0.1
    X_WP10 = 0.1
else:
    X_WP1 = 0.2
    X_WP2 = 0.2
    X_WP3 = 0.2
    X_WP4 = 0.1
    X_WP5 = 0.1
    X_WP6 = 0.1
    X_WP7 = 0.0
    X_WP8 = 0.0
    X_WP9 = 0.0
    X_WP10 = 0.1

if round(X_WP1 + X_WP2 + X_WP3 + X_WP4 + X_WP5 + X_WP6 + X_WP7 + X_WP8 + X_WP9 + X_WP10,1) == 1:
    bereken_massadebieten_in_leidingen()
else:
    print("Massafracties door warmtepompen zijn samen niet gelijk aan 1")
    exit()



parameters = ["selected_model","delta_T","m","T_in_source","T_out_source","T_building","T_req_building","Q_building","Q_req_building", "percentage", "P_compressor","snelheid","diameter","lengte"]
for i in range(1, 11):
    WP = "WP"+str(i)
    for j in range(len(parameters)):
        name_var = parameters[j] + "_" + WP
        value = globals().get(name_var)
        warmtepompen[WP][parameters[j]] = value

for i in range(1, 11):
    WP = "WP"+str(i)
    warmtepompen_state[WP]['state'] = 'aan'
    warmtepompen_state[WP]['delta_T_onthoud'] = warmtepompen[WP]['delta_T']


##############################

### ALGEMENE WERKING SCRIPT
initial_guess_T_WW_in = 10
iteratie_error_marge = 0.001
aantal_cijfers_na_komma = 2
warmtepompen_simulatie ={}
drukverliezen_tot = []
pompvermogen_tot = []
####################################
###### START SCRIPT ################
####################################
for WP in warmtepompen:
    if isinstance(warmtepompen[WP]['Q_req_building'], list):
        warmtepompen_kopie[WP]['Q_req_building'] = warmtepompen[WP]['Q_req_building'][:]
        warmtepompen_kopie[WP]['T_req_building'] = warmtepompen[WP]['T_req_building'][:]
    else:
        warmtepompen_kopie[WP]['Q_req_building'] = warmtepompen[WP]['Q_req_building']
        warmtepompen_kopie[WP]['T_req_building'] = warmtepompen[WP]['T_req_building']

if keuze_simulatie == 'SS':
    indices = [0]
else:
    indices = range(len(temperatuur))
huidige_iteratie = None
progress_bar = st.progress(0)
for i in indices:
    drukverliezen = 0
    pompvermogen = 0
    scaled_i = 1+int((i - 0) / (len(indices) - 0) * 99)
    progress_bar.progress(scaled_i)
    for WP in warmtepompen:
        if isinstance(warmtepompen_kopie[WP]['Q_req_building'], list):
            warmtepompen[WP]['Q_req_building'] = warmtepompen_kopie[WP]['Q_req_building'][i]
            warmtepompen[WP]['T_req_building'] = warmtepompen_kopie[WP]['T_req_building'][i]
        else:
            def enkel_waarde(x):
                return x[0] if isinstance(x, list) else x
            warmtepompen[WP]['Q_req_building'] = enkel_waarde(warmtepompen_kopie[WP]['Q_req_building'])
            warmtepompen[WP]['T_req_building'] = enkel_waarde(warmtepompen_kopie[WP]['T_req_building'])
    solution = {}
    T_16 = initial_guess_T_WW_in
    T_16_old = initial_guess_T_WW_in - iteratie_error_marge - 1
    eerste_keer_berekend = False
    if i == 0:
        a = 0
    elif i % 24 == 0:
        a += 1

    T_ground = grond_temperatuur[a]
    while abs(T_16 - T_16_old) > iteratie_error_marge:

        for i in range(1, 11):
            warmtepompen["WP" + str(i)]["drukverlies"] = 0
        for i in range(1, 8):
            leidingen[i]["drukverlies"] = 0



        C_imec = Cp_fluid_imec * m_dot_imec
        C_backbone = Cp_fluid_backbone * m_dot_backbone
        C_min = min(C_imec, C_backbone)
        C_max = max(C_imec, C_backbone)
        Q_max = C_min * (T_imec - T_16)
        C_star = C_min / C_max
        NTU = U * A / C_min

        if type_WW == 'gelijkstroom':
            epsilon = (1 - math.exp(-NTU * (1 + C_star))) / (1 + C_star)
        elif type_WW == 'tegenstroom':
            epsilon = (1 - math.exp(-NTU * (1 - C_star))) / (1 - C_star * math.exp(-NTU * (1 - C_star)))
        else:
            epsilon = 0

        # andere optie
        # epsilon=(1 / C_star) * (1 - np.exp(-C_star * (1 - np.exp(-NTU))))

        Q = epsilon * Q_max
        T_16_old = T_16
        T_naar_Dijle = T_imec - Q / (m_dot_imec * Cp_fluid_imec)
        T_1 = T_16 + Q / (m_dot_backbone * Cp_fluid_backbone)

        T_16 = T_1 - T_daling_totaal(T_1,T_16,m_dot_backbone,T_ground)

        m_dot_backbone = update_massadebieten(m_dot_backbone)

        for i in range(1, 11):
            keuze = globals().get(f"keuze_WP{i}")
            wp_id = f'WP{i}'
            if warmtepompen[wp_id]['model'] != 'fixed':  # of gebruik model_WP_i als het per WP verschilt
                if keuze == "Snelheid":
                    snelheid = globals().get(f"snelheid_WP{i}")
                    if snelheid is not None:
                        diameter = bereken_diameter(wp_id, snelheid)
                        warmtepompen[wp_id]['diameter'] = diameter
                elif keuze == "Diameter":
                    diameter = globals().get(f"diameter_WP{i}")
                    if diameter is not None:
                        snelheid = bereken_snelheid(wp_id, diameter)
                        warmtepompen[wp_id]['snelheid'] = snelheid

        range_vals = range(1, 8) if WP7_8_9_checkbox else range(1, 7)
        for i in range_vals:
            if keuze_hot[i] == "Snelheid":
                snelheid = leidingen[i]['hot']['snelheid']
                diameter = bereken_diameter_pipe(i, snelheid, 'hot')
                leidingen[i]['hot']['diameter'] = diameter
            elif keuze_hot[i] == "Diameter":
                diameter = leidingen[i]['hot']['diameter']
                snelheid = bereken_snelheid_pipe(i, diameter, 'hot')
                leidingen[i]['hot']['snelheid'] = snelheid

            if keuze_cold[i] == "Snelheid":
                snelheid = leidingen[i]['cold']['snelheid']
                diameter = bereken_diameter_pipe(i, snelheid, 'cold')
                leidingen[i]['cold']['diameter'] = diameter
            elif keuze_cold[i] == "Diameter":
                diameter = leidingen[i]['cold']['diameter']
                snelheid = bereken_snelheid_pipe(i, diameter, 'cold')
                leidingen[i]['cold']['snelheid'] = snelheid
    paden = []
    paden.append(leidingen[1]["drukverlies"] + leidingen[2]["drukverlies"] + warmtepompen["WP1"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[2]["drukverlies"] + warmtepompen["WP2"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + warmtepompen["WP3"]["drukverlies"])
    paden.append(
        leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + warmtepompen["WP4"][
            "drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + leidingen[5][
        "drukverlies"] + warmtepompen["WP5"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + leidingen[5][
        "drukverlies"] + leidingen[6]["drukverlies"] + warmtepompen["WP6"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + leidingen[5][
        "drukverlies"] + leidingen[6]["drukverlies"] + leidingen[7]["drukverlies"] + warmtepompen["WP7"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + leidingen[5][
        "drukverlies"] + leidingen[6]["drukverlies"] + leidingen[7]["drukverlies"] + warmtepompen["WP8"]["drukverlies"])
    paden.append(leidingen[1]["drukverlies"] + leidingen[3]["drukverlies"] + leidingen[4]["drukverlies"] + leidingen[5][
        "drukverlies"] + leidingen[6]["drukverlies"] + leidingen[7]["drukverlies"] + warmtepompen["WP9"]["drukverlies"])

    pompvermogen_tot.append((max(paden) * m_dot_backbone) / (dichtheid_fluid_backbone * 0.7))


    wp_ids = [f'WP{i}' for i in range(1, 7)] + ['WP10']
    if WP7_8_9_checkbox:
        wp_ids += [f'WP{i}' for i in range(7, 10)]
    for wp_id in wp_ids:
        if warmtepompen[wp_id]['Q_req_building'] == 0:
            warmtepompen[wp_id]['T_req_building'] = 0
        wp = warmtepompen[wp_id]
        wp.update({
            'Q_boiler': wp['Q_req_building'] - wp['Q_building'],
            'T_boiler': wp['T_req_building'] - wp['T_building']
        })

    solution['T WARMTEWISSELAAR OUT'] = T_1
    solution['T WARMTEWISSELAAR IN'] = T_16
    solution['T naar Dijle'] = T_naar_Dijle

###############################

    for Temperatuur in solution:
        solution[Temperatuur] = round(solution[Temperatuur], aantal_cijfers_na_komma)
    for WP in warmtepompen.keys():
        if not WP7_8_9_checkbox and WP in ["WP7", "WP8", "WP9"]:
            continue
        warmtepompen[WP]['T_in_source'] = round(warmtepompen[WP]['T_in_source'], aantal_cijfers_na_komma)
        warmtepompen[WP]['T_out_source'] = round(warmtepompen[WP]['T_out_source'], aantal_cijfers_na_komma)

    update_simulatie_dict(warmtepompen, warmtepompen_simulatie)

##############################################################

####################################
###### SHOW SOLUTION ###############
####################################
#st.title("WARMTENET")
st.pyplot(teken_schema(solution))
toon_ontwerpparameters()

import pickle
# Verpak alles wat je wil opslaan in een dictionary
save_dict = {
    "warmtepompen_simulatie": warmtepompen_simulatie,
    "WP7_8_9_checkbox": WP7_8_9_checkbox,
    #"drukverliezen": drukverliezen_tot,
    "pompvermogen": pompvermogen_tot
}

with open("data.pkl", "wb") as f:
    pickle.dump(save_dict, f)

st.success("Alles is berekend en opgeslagen")
#         streamlit run Main.py
