import pandas as pd
import numpy as np
import xlwings as xw

from .data import HOLDDOWN_DF, ROD_DATA, SHEAR_CLIP_DATA_TABLE, BEARING_PLATE_DATA
from .calculations import *

def indv_shear_wall_analysis(workbook_path=None):
    #___________________________________________________
    # Get the name of the Excel file
    file_name = workbook_path or xw.Book.caller().fullname
    wb = xw.Book(file_name)
    ws = wb.sheets.active

    df = ws['C9:Z17'].options(pd.DataFrame).value
    # df
    
 

    # ------------------------------------------------------------------------------
    # Convert Bearing Plate Data to DataFrame
    # ------------------------------------------------------------------------------
    df_bearing_plate_data = pd.DataFrame(BEARING_PLATE_DATA)

 

    # ------------------------------------------------------------------------------
    # Read Storey and Geometry Information from Excel (Dynamic Range)
    # ------------------------------------------------------------------------------

    # Get number of storeys from cell F6
    no_of_storey = int(ws.range('F6').value)
    print("No. of storeys:", no_of_storey)

    # Define start row and calculate end row based on no_of_storey
    start_row_geom = 11
    start_row_force = 27
    end_row_geom = start_row_geom + no_of_storey - 1
    end_row_force = start_row_force + no_of_storey - 1

    # Construct dynamic ranges for each input
    shear_wall_height = ws.range(f'D{start_row_geom}:D{end_row_geom}').value   # Hi, m
    shear_wall_length = ws.range(f'E{start_row_geom}:E{end_row_geom}').value   # Ls, m 
    shear_force = ws.range(f'D{start_row_force}:D{end_row_force}').value       # Force, kN

    # Convert to NumPy arrays
    Force_list = np.array(shear_force)
    Cumulative_force_list = np.cumsum(Force_list)

    shear_wall_height_list = np.array(shear_wall_height)
    shear_wall_length_list = np.array(shear_wall_length)





    # ------------------------------------------------------------------------------
    # Read Shear Wall Geometry and Material Properties from Excel (Dynamic Range)
    # ------------------------------------------------------------------------------

    # Base row for property data (vertical blocks)
    start_row_material = 41
    end_row_material = start_row_material + no_of_storey - 1

    # Panel, nail, and shear wall properties
    panel_type_list = np.array(ws.range(f'C{start_row_material}:C{end_row_material}').value)
    panel_thickness_list = np.array(ws.range(f'D{start_row_material}:D{end_row_material}').value)
    panel_side_list = np.array(ws.range(f'E{start_row_material}:E{end_row_material}').value)
    nail_length_list = np.array(ws.range(f'F{start_row_material}:F{end_row_material}').value)
    nail_diameter_list = np.array(ws.range(f'G{start_row_material}:G{end_row_material}').value)
    nail_spacing_list = np.array(ws.range(f'H{start_row_material}:H{end_row_material}').value)

    # Additional mechanical parameters
    t2_data_list = np.array(ws.range(f'Y{start_row_material - 1}:Y{end_row_material - 1}').value)
    a_data_list = np.array(ws.range(f'Z{start_row_material - 1}:Z{end_row_material - 1}').value)
    b_data_list = np.array(ws.range(f'AA{start_row_material - 1}:AA{end_row_material - 1}').value)
    phi_data_list = np.array(ws.range(f'AB{start_row_material - 1}:AB{end_row_material - 1}').value)
    K_D_data_list = np.array(ws.range(f'AC{start_row_material - 1}:AC{end_row_material - 1}').value)
    K_SF_data_list = np.array(ws.range(f'AD{start_row_material - 1}:AD{end_row_material - 1}').value)
    K_T_data_list = np.array(ws.range(f'AE{start_row_material - 1}:AE{end_row_material - 1}').value)
    K_s_data_list = np.array(ws.range(f'AF{start_row_material - 1}:AF{end_row_material - 1}').value)
    J_x_data_list = np.array(ws.range(f'AG{start_row_material - 1}:AG{end_row_material - 1}').value)
    J_D_data_list = np.array(ws.range(f'AH{start_row_material - 1}:AH{end_row_material - 1}').value)
    n_s_data_list = np.array(ws.range(f'AI{start_row_material - 1}:AI{end_row_material - 1}').value)
    J_us_data_list = np.array(ws.range(f'AJ{start_row_material - 1}:AJ{end_row_material - 1}').value)
    J_hd_data_list = np.array(ws.range(f'AK{start_row_material - 1}:AK{end_row_material - 1}').value)
    J_s_data_list = np.array(ws.range(f'AL{start_row_material - 1}:AL{end_row_material - 1}').value)
    G_data_list = np.array(ws.range(f'AM{start_row_material - 1}:AM{end_row_material - 1}').value)
    G_f3_data_list = np.array(ws.range(f'AN{start_row_material - 1}:AN{end_row_material - 1}').value)
    joist_depth_list = np.array(ws.range(f'AZ{start_row_material - 1}:AZ{end_row_material - 1}').value)
    floor_sheathing_list = np.array(ws.range(f'BA{start_row_material - 1}:BA{end_row_material - 1}').value)

    # Holddown and stud info
    tud_data_list = np.array(ws.range(f'N94:N{93 + no_of_storey}').value)
    holddown_type_list = np.array(ws.range(f'L{start_row_material}:L{end_row_material}').value)
    holddown_model_list = np.array(ws.range(f'M{start_row_material}:M{end_row_material}').value)
    holddown_quantity_list = np.array(ws.range(f'N{start_row_material}:N{end_row_material}').value)
    stud_type_list = np.array(ws.range(f'R{start_row_material}:R{end_row_material}').value)
    no_of_stud_list = np.array(ws.range(f'S{start_row_material}:S{end_row_material}').value) * 2

    # Species and grading
    species_list = np.array(ws.range(f'AO{start_row_material - 1}:AO{end_row_material - 1}').value)
    species_list_tolist = species_list.tolist()
    species_grade_list = np.array(ws.range(f'AP{start_row_material - 1}:AP{end_row_material - 1}').value)
    species_grade_list_tolist = species_grade_list.tolist()

    # Shear clip info and bearing plate types
    shear_clip_data_list = np.array(ws.range(f'E69:E{68 + no_of_storey}').value)
    bp_type_data_list = np.array(ws.range(f'L94:L{93 + no_of_storey}').value)


    # Unit conversions for force and geometry
    V_i_list_N = Cumulative_force_list * 1000
    H_i_list_mm = shear_wall_height_list * 1000
    L_i_list_mm = shear_wall_length_list * 1000
    stud_height_mm = H_i_list_mm-joist_depth_list-floor_sheathing_list-(3*38)

    # Shear stress calculation
    ve_list = Cumulative_force_list / shear_wall_length_list




    # ------------------------------------------------------------------------------
    # Load Information from Excel (Dynamic Range)
    # ------------------------------------------------------------------------------

    # Calculate dynamic end row based on no_of_storey
    start_row_load = 27
    end_row_load = start_row_load + no_of_storey - 1

    # Read uniform dead and live loads
    uniform_dead_list = np.array(ws.range(f'G{start_row_load}:G{end_row_load}').value)
    uniform_live_list = np.array(ws.range(f'H{start_row_load}:H{end_row_load}').value)
    print("Uniform dead loads:", uniform_dead_list)
    print("Uniform live loads:", uniform_live_list)

    # Dead and live load cumulative
    dead_load = uniform_dead_list * shear_wall_length_list
    dead_load_cum = np.cumsum(dead_load)
    print("Cumulative dead load:", dead_load_cum)

    live_load = uniform_live_list * shear_wall_length_list
    live_load_cum = np.cumsum(live_load)
    print("Cumulative live load:", live_load_cum)

    # ------------------------------------------------------------------------------
    # Wall Self-Weight from Excel (Dynamic Range)
    # ------------------------------------------------------------------------------

    start_row_weight = 276
    end_row_weight = start_row_weight + no_of_storey - 1

    wall_weight_list = np.array(ws.range(f'D{start_row_weight}:D{end_row_weight}').value)
    print("Wall weights:", wall_weight_list)


    # ------------------------------------------------------------------------------
    # Moment Calculation per Storey
    # ------------------------------------------------------------------------------
    Moment_Lists = np.zeros_like(Force_list)
    for i in range(len(Force_list)):
        for j in range(i + 1):
            Moment_Lists[i] += Force_list[j] * np.sum(shear_wall_height_list[j:i + 1])
    print("Moment list:", Moment_Lists)

    # ------------------------------------------------------------------------------
    # Load Case and Force Calculations (Tf and Cf)
    # ------------------------------------------------------------------------------

    user_factor_input = ws['K21'].value
    selected_case = ws['F21'].value.strip() if ws['F21'].value else ""

    if selected_case in ["Flexible diaphragm wind", "Rigid diaphragm wind", "Envelope diaphragm wind"]:
        factor = user_factor_input
    else:
        factor = user_factor_input

    # ws['H21'].value = f"Applied load factor: {factor}"

    Tf_list = factor * ((Moment_Lists / shear_wall_length_list) - (dead_load_cum / 2))
    Cf_list = factor * ((Moment_Lists / shear_wall_length_list) + ((dead_load_cum + 0.5 * live_load_cum) / 2))

    print("Tf_list:", Tf_list)
    print("Cf_list:", Cf_list)

    

    # ------------------------------------------------------------------------------
    # Sheathing-to-Framing Strength and Buckling Calculations for Each Storey
    # ------------------------------------------------------------------------------

    # Initialize empty lists for intermediate and final values
    n_u_list = []                # Unit lateral strength resistance
    N_u_list = []                # Adjusted lateral strength
    V_rs_list = []              # Sheathing-to-framing shear strength
    alpha_list = []             # Buckling parameter alpha
    eta_list = []               # Buckling parameter eta
    k_pb_list = []              # Buckling coefficient
    v_pb_list = []              # Buckling shear
    V_rsbuckling_list = []      # Panel buckling shear strength
    v_rs_govern_list = []       # Governing shear value (min of above two)

    # Loop through each storey using the number of storeys
    for i in range(no_of_storey):
        # Calculate unit lateral strength resistance
        n_u = unit_lateral_strength_resistance(
            nail_diameter_list[i],
            panel_thickness_list[i],
            t2_data_list[i],
            J_x_data_list[i],
            G_data_list[i],
            G_f3_data_list[i]
        )
        n_u_list.append(n_u)

        # Adjusted lateral strength
        N_u = n_u * K_D_data_list[i] * K_SF_data_list[i] * K_T_data_list[i]
        N_u_list.append(N_u)

        # Sheathing-to-framing shear resistance
        V_rs = sheathing_to_framing_shear(
            phi_data_list[i],
            N_u,
            nail_spacing_list[i],
            J_D_data_list[i],
            n_s_data_list[i],
            J_us_data_list[i],
            J_s_data_list[i],
            J_hd_data_list[i]
        )
        V_rs_list.append(V_rs)

        # Get material properties based on panel thickness
        thickness_data = get_data_by_thickness(panel_thickness_list[i])
        B_v = thickness_data.Bv
        B_a0 = thickness_data.Ba_0
        B_a90 = thickness_data.Ba_90

        # Buckling properties
        alpha, eta, k_pb, v_pb = panel_buckling_strength(
            a_data_list[i],
            b_data_list[i],
            panel_thickness_list[i],
            B_v,
            B_a0,
            B_a90
        )
        alpha_list.append(alpha)
        eta_list.append(eta)
        k_pb_list.append(k_pb)
        v_pb_list.append(v_pb)

        # Buckling shear strength
        V_rsbuckling = panel_buckling_shear(
            phi_data_list[i],
            K_D_data_list[i],
            K_s_data_list[i],
            K_T_data_list[i],
            v_pb
        )
        V_rsbuckling_list.append(V_rsbuckling)

        # Governing value (sheathing-to-framing or buckling)
        v_rs_govern = round(min(V_rs, V_rsbuckling), 2)
        multiplier = 2 if panel_side_list[i] == "B.S" else 1
        v_rs_govern_list.append(v_rs_govern * multiplier)

        # ------------------------------------------------------------------------------
        # Convert lists to NumPy arrays for downstream usage
        # ------------------------------------------------------------------------------
        nu_array = np.array(n_u_list)
        N_u_array = np.array(N_u_list)
        V_rs_array = np.array(V_rs_list)
        alpha_array = np.array(alpha_list)
        eta_array = np.array(eta_list)
        k_pb_array = np.array(k_pb_list)
        v_pb_array = np.array(v_pb_list)
        V_rsbuckling_array = np.array(V_rsbuckling_list)
        v_rs_govern_array = np.array(v_rs_govern_list)


    # ------------------------------------------------------------------------------
    # Shear Clip Capacity Lookup
    # ------------------------------------------------------------------------------

    # List to store factored resistance values for shear clips
    factored_resistance_values = []

    # Ensure we are only looping through storey-specific data
    for i in range(no_of_storey):
        stud_species = species_list_tolist[i]
        clip_type = shear_clip_data_list[i]

        # Match stud species and clip type to fetch resistance
        for item in SHEAR_CLIP_DATA_TABLE:
            if item["Stud Species"] == stud_species:
                factored_resistance_values.append(item.get(clip_type, None))
                break  # Stop after first match

    # Print and convert to NumPy array
    print("Factored Resistance values (kN):", factored_resistance_values)
    factored_resistance_array = np.array(factored_resistance_values)
    print("Factored Resistance Array:", factored_resistance_array)


    # Deflection calculation section
    
    
    # -----------------------------------------------------------------------------------
    # Main Deflection Calculation Logic (Dynamic to no_of_storey)
    # -----------------------------------------------------------------------------------

    panel_type_list_tolist = panel_type_list.tolist()
    panel_thickness_list_tolist = panel_thickness_list.tolist()

    # Get bv values for each storey depending on panel type/thickness
    bv_list = []
    for i, (panel_type, thickness) in enumerate(zip(panel_type_list_tolist, panel_thickness_list_tolist)):
        bv = get_bv(panel_type, thickness)
        bv_list.append(bv * 2 if panel_side_list[i] == "B.S" else bv)

    # Elasticity in parallel and perpendicular directions
    epar_list, eperp_list = [], []
    for i in range(no_of_storey):
        epar, eperp = get_elasticity(species_list[i], species_grade_list[i])
        epar_list.append(epar)
        eperp_list.append(eperp)
    epar_list_array = np.array(epar_list)

    # Modulus of elasticity for threaded rods (Et)
    Et_list = [200000 for h in holddown_type_list[:no_of_storey] if h == 'Rod']
    Et_list_array = np.array(Et_list)

    # Cross-sectional area for stud configurations
    A_2x6, A_2x4 = 38 * 140, 38 * 89
    Ac_list = []
    for stud_type, n in zip(stud_type_list[:no_of_storey], no_of_stud_list[:no_of_storey]):
        A = A_2x6 if stud_type == '2x6' else A_2x4 if stud_type == '2x4' else 0
        Ac_list.append(A * n)
    Ac_list_array = np.array(Ac_list)

    # Rod parameters from rod database
    ag_values, ane_values, A_t_values, tr_values = [], [], [], []
    for model in holddown_model_list[:no_of_storey]:
        for rod in ROD_DATA:
            if model in (rod["Strong Rod Standard"], rod["Strong Rod High Strength"]):
                ag_values.append(rod["Ag"])
                ane_values.append(rod["Ane"])
                A_t_values.append(0.4 * rod["Ag"] + 0.6 * rod["Ane"])
                tr_values.append(rod["Tr Standard"] if model == rod["Strong Rod Standard"] else rod["Tr High"])
                break
    At_list_array = np.array(A_t_values)

    # Combined stiffness and inertia
    n_list = Et_list_array / epar_list_array
    At_tr_list = At_list_array * n_list
    Lc_list = []

    symmetry_case = ws['T25'].value.strip() if ws['T25'].value else ""
    for L, n_stud in zip(shear_wall_length_list, no_of_stud_list[:no_of_storey]):
        if symmetry_case == "symmetric":
            Lc = L * 1000 - (n_stud + 228)
        elif symmetry_case.startswith("Asym_"):
            m = int(symmetry_case.split("_")[1])
            Lc = L * 1000 - (m*38 + 9*25)
        else:
            Lc = L * 1000 - (4*38 + 9*25)
        Lc_list.append(Lc)
    Lc_list_array = np.array(Lc_list)

    # Moment of inertia and flexural rigidity
    y_tr_list = (Ac_list_array * Lc_list_array) / (Ac_list_array + At_tr_list)
    I_tr_list = (At_tr_list * y_tr_list**2) + (Ac_list_array * (Lc_list_array - y_tr_list)**2)
    EI_tr_list = epar_list_array * I_tr_list

    # Bending moment and deflection
    M_list = []
    M_cumulative = [0]
    for i in range(1, no_of_storey + 1):
        M = V_i_list_N[i - 1] * H_i_list_mm[i - 1]
        M_list.append(M)
        M_cumulative.append(M_cumulative[-1] + M)
    M_cumulative = M_cumulative[:no_of_storey]

    D_bending_list = [
        deflection_due_to_bending(V_i_list_N[i], H_i_list_mm[i], EI_tr_list[i], M_cumulative[i])
        for i in range(no_of_storey)
    ]

    # Panel shear deflection
    D_panel_shear_list = [
        deflection_due_to_panel_shear(V_i_list_N[i], H_i_list_mm[i], L_i_list_mm[i], bv_list[i])
        for i in range(no_of_storey)
    ]

    # Nail slip calculation
    v_list = V_i_list_N / L_i_list_mm
    v_s_list = [
        (v_list[i] * nail_spacing_list[i]) / (2 if panel_side_list[i] == "B.S" else 1)
        for i in range(no_of_storey)
    ]
    v_s_array = np.array(v_s_list)
    en_list = ((0.013 * v_s_array) / nail_diameter_list[:no_of_storey]**2) ** 2
    D_nail_slip = 0.0025 * H_i_list_mm * en_list

    # Wall anchorage deflection
    seating_increments_mm, deflections_mm = [], []
    for model_no in tud_data_list[:no_of_storey]:
        match = HOLDDOWN_DF[HOLDDOWN_DF['Model No.'] == model_no]
        if not match.empty:
            seating_increments_mm.append(match['Seating Increment (DR) (mm)'].values[0])
            deflections_mm.append(match['Deflection at Factored Resistance (DF) (mm)'].values[0])


    # Initialize result list
    bp_factored_resistances = []

    # Loop dynamically over each storey
    for i in range(no_of_storey):
        model_no = bp_type_data_list[i]
        species_type = species_list_tolist[i]

        for item in BEARING_PLATE_DATA:
            if item["Model No."] == model_no:
                if species_type == "D.Fir-L":
                    bp_factored_resistances.append(item["D.Fir-L (kN)"])
                elif species_type == "S-P-F":
                    bp_factored_resistances.append(item["S-P-F (kN)"])
                break  # Stop at the first match




    # Initialize list to store compressive resistance
    cr_list = []

    # # Loop through each storey dynamically
    # for i in range(no_of_storey):
    #     stud_depth = 140 if stud_type_list[i] == '2x6' else 89
    #     cr = factored_compressive_resistance(
    #         no_of_stud_list[i], 
    #         stud_depth, 
    #         H_i_list_mm[i]
    #     )
    #     cr_list.append(cr)

    # Loop through each storey dynamically
    for i in range(no_of_storey):
        stud_depth = 140 if stud_type_list[i] == '2x6' else 89
        cr = factored_compressive_resistance(
            no_of_stud_list[i], 
            stud_depth, 
            stud_height_mm[i]
        )
        cr_list.append(cr)




    L_c_plates = [3 * 38 for _ in range(no_of_storey)]
    Cf_list = np.array(Cf_list)
    eperp_list = np.array(eperp_list)
    Ac_list = np.array(Ac_list)
    d_a_list = (Tf_list / tr_values) * deflections_mm + seating_increments_mm + (Cf_list * 1000 / (eperp_list * Ac_list)) * L_c_plates
    D_wall_anchorage_list = deflection_due_to_wall_anchorage(np.array(d_a_list), H_i_list_mm, L_i_list_mm)

    # Bottom rotation deflection
    theta_list = [
        (M_cumulative[i]*H_i_list_mm[i])/EI_tr_list[i] +
        (V_i_list_N[i]*H_i_list_mm[i]**2)/(2*EI_tr_list[i])
        for i in range(no_of_storey)
    ]
    theta_cumsum = np.cumsum(theta_list[::-1])[::-1]
    alpha_cumsum = np.cumsum((np.array(d_a_list) / L_i_list_mm)[::-1])[::-1]
    D_rotation_bottom_list = [
        deflection_due_to_rotation_at_bottom(H_i_list_mm[i], theta_cumsum[i], alpha_cumsum[i])
        for i in range(no_of_storey)
    ]

    # Final total deflection
    D_total_list = [
        total_deflection(D_bending_list[i], D_panel_shear_list[i], D_nail_slip[i],
                        D_wall_anchorage_list[i], D_rotation_bottom_list[i])
        for i in range(no_of_storey)
    ]
    D_total_list_np = np.array(D_total_list)


    #-----------------------------------------------------------------------------------    
    # Deflection Calculations
    # According to Clause 4.1.3.5-(3) - the total drift per storey under service wind and gravity shall not exceed 1/500 of the storey height 0.20%
    # According to Clause 4.1.8.13 -(2) - Lateral deflections shall be multiplied by RdRo/Ie to increase and give realistic values of anticipated deflections 
    # According to Clause 4.1.8.13 - (3) - the largest interstorey deflection shall be limited to 0.025*hs or 1/40 of the storey height. 2.5%
    #-----------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # Determine Drift Limit Based on Load Case
    # ------------------------------------------------------------------------------

    drift_limit = 0.0020 if selected_case in [
        "Flexible diaphragm wind",
        "Rigid diaphragm wind",
        "Envelope diaphragm wind"
    ] else 0.025

    # Create a list of constant drift limits per storey
    drift_limit_list = [drift_limit] * no_of_storey

    # Write to Excel dynamically based on no_of_storey
    start_row_drift = 82
    end_row_drift = start_row_drift + no_of_storey - 1
    ws.range(f"V{start_row_drift}:V{end_row_drift}").options(transpose=True).value = drift_limit_list
    ws['V80'].value = "Wind" if drift_limit == 0.0020 else "Seismic"

    # ------------------------------------------------------------------------------
    # Calculate Fundamental Period of the Wall
    # ------------------------------------------------------------------------------

    t_upper = wall_weight_list * D_total_list_np**2
    t_lower = Force_list * D_total_list_np

    print("t_upper:", t_upper)
    print("t_lower:", t_lower)

    fundamental_period = 2 * math.pi * sqrt(np.sum(t_upper) / (9800 * np.sum(t_lower)))
    print("Fundamental Period:", fundamental_period)

    # ------------------------------------------------------------------------------
    # Wall Stiffness Calculation
    # ------------------------------------------------------------------------------

    stiffness_list = []
    for i in range(no_of_storey):
        comp1 = (2 * v_list[i] * H_i_list_mm[i]**2) / (3 * epar_list[i] * Ac_list[i] * L_i_list_mm[i])
        comp2 = (v_list[i] * H_i_list_mm[i]) / bv_list[i]
        comp3 = 0.0025 * H_i_list_mm[i] * en_list[i]
        comp4 = (H_i_list_mm[i] * d_a_list[i]) / L_i_list_mm[i]
        stiffness = (v_list[i] * L_i_list_mm[i]) / (comp1 + comp2 + comp3 + comp4)
        stiffness_list.append(stiffness)

    average_stiffness = np.average(stiffness_list)
    print("Storey-wise Stiffness:", stiffness_list)
    print("Average Wall Stiffness:", average_stiffness)




    # ------------------------------------------------------------------------------
    # All output to Excel (Dynamic Ranges)
    # ------------------------------------------------------------------------------

    def get_excel_range(col_letter, start_row, num_rows):
        end_row = start_row + num_rows - 1
        return f"{col_letter}{start_row}:{col_letter}{end_row}"

    # Dynamic ranges
    load_summary_start = 27
    vr_table_start = 41
    conn_table_start = 129

    # Load summary table values
    ws.range(get_excel_range('E', load_summary_start, no_of_storey)).options(transpose=True).value = Cumulative_force_list
    ws.range(get_excel_range('F', load_summary_start, no_of_storey)).options(transpose=True).value = ve_list
    ws.range(get_excel_range('I', load_summary_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list
    ws.range(get_excel_range('J', load_summary_start, no_of_storey)).options(transpose=True).value = dead_load_cum
    ws.range(get_excel_range('K', load_summary_start, no_of_storey)).options(transpose=True).value = live_load_cum
    ws.range(get_excel_range('L', load_summary_start, no_of_storey)).options(transpose=True).value = Moment_Lists
    ws.range(get_excel_range('M', load_summary_start, no_of_storey)).options(transpose=True).value = Tf_list
    ws.range(get_excel_range('N', load_summary_start, no_of_storey)).options(transpose=True).value = Cf_list

    # Shear capacity Vr values
    ws.range(get_excel_range('I', vr_table_start, no_of_storey)).options(transpose=True).value = v_rs_govern_array
    ws.range(get_excel_range('O', vr_table_start, no_of_storey)).options(transpose=True).value = tr_values
    ws.range(get_excel_range('P', vr_table_start, no_of_storey)).options(transpose=True).value = Tf_list
    ws.range(get_excel_range('T', vr_table_start, no_of_storey)).options(transpose=True).value = cr_list
    ws.range(get_excel_range('U', vr_table_start, no_of_storey)).options(transpose=True).value = Cf_list

    # Sheathing to framing connection table values
    ws.range(get_excel_range('D', conn_table_start, no_of_storey)).options(transpose=True).value = N_u_array / nail_spacing_list
    ws.range(get_excel_range('E', conn_table_start, no_of_storey)).options(transpose=True).value = N_u_array
    ws.range(get_excel_range('F', conn_table_start, no_of_storey)).options(transpose=True).value = nail_spacing_list
    ws.range(get_excel_range('G', conn_table_start, no_of_storey)).options(transpose=True).value = J_D_data_list
    ws.range(get_excel_range('H', conn_table_start, no_of_storey)).options(transpose=True).value = n_s_data_list
    ws.range(get_excel_range('I', conn_table_start, no_of_storey)).options(transpose=True).value = J_us_data_list
    ws.range(get_excel_range('J', conn_table_start, no_of_storey)).options(transpose=True).value = J_s_data_list
    ws.range(get_excel_range('K', conn_table_start, no_of_storey)).options(transpose=True).value = J_hd_data_list
    ws.range(get_excel_range('L', conn_table_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list



   # Define the helper function if not already defined
    def get_excel_range(col_letter, start_row, num_rows):
        end_row = start_row + num_rows - 1
        return f"{col_letter}{start_row}:{col_letter}{end_row}"

    # Starting rows for each table
    buckling_table_start = 151
    bending_deflection_start = 196
    bending_prop_start = 205
    panel_shear_deflection_start = 219

    # --------------------------------------------------------------------
    # Sheathing panel buckling table values back to Excel 
    # --------------------------------------------------------------------
    ws.range(get_excel_range('D', buckling_table_start, no_of_storey)).options(transpose=True).value = v_pb_array
    ws.range(get_excel_range('E', buckling_table_start, no_of_storey)).options(transpose=True).value = K_D_data_list
    ws.range(get_excel_range('F', buckling_table_start, no_of_storey)).options(transpose=True).value = K_s_data_list
    ws.range(get_excel_range('G', buckling_table_start, no_of_storey)).options(transpose=True).value = K_T_data_list
    ws.range(get_excel_range('H', buckling_table_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list
    ws.range(get_excel_range('I', buckling_table_start, no_of_storey)).options(transpose=True).value = alpha_array
    ws.range(get_excel_range('J', buckling_table_start, no_of_storey)).options(transpose=True).value = eta_array
    ws.range(get_excel_range('K', buckling_table_start, no_of_storey)).options(transpose=True).value = k_pb_array

    # --------------------------------------------------------------------
    # 1. Deflection due to bending 
    # --------------------------------------------------------------------
    ws.range(get_excel_range('D', bending_deflection_start, no_of_storey)).options(transpose=True).value = Cumulative_force_list * 1000  # Vi, N
    ws.range(get_excel_range('E', bending_deflection_start, no_of_storey)).options(transpose=True).value = shear_wall_height_list * 1000  # Hi, mm
    ws.range(get_excel_range('F', bending_deflection_start, no_of_storey)).options(transpose=True).value = M_cumulative
    ws.range(get_excel_range('G', bending_deflection_start, no_of_storey)).options(transpose=True).value = EI_tr_list
    ws.range(get_excel_range('H', bending_deflection_start, no_of_storey)).options(transpose=True).value = D_bending_list

    # Bending properties
    ws.range(get_excel_range('D', bending_prop_start, no_of_storey)).options(transpose=True).value = epar_list
    ws.range(get_excel_range('E', bending_prop_start, no_of_storey)).options(transpose=True).value = Et_list
    ws.range(get_excel_range('F', bending_prop_start, no_of_storey)).options(transpose=True).value = Ac_list
    ws.range(get_excel_range('G', bending_prop_start, no_of_storey)).options(transpose=True).value = A_t_values
    ws.range(get_excel_range('H', bending_prop_start, no_of_storey)).options(transpose=True).value = n_list
    ws.range(get_excel_range('I', bending_prop_start, no_of_storey)).options(transpose=True).value = At_tr_list
    ws.range(get_excel_range('J', bending_prop_start, no_of_storey)).options(transpose=True).value = Lc_list_array
    ws.range(get_excel_range('K', bending_prop_start, no_of_storey)).options(transpose=True).value = y_tr_list
    ws.range(get_excel_range('L', bending_prop_start, no_of_storey)).options(transpose=True).value = I_tr_list

    # --------------------------------------------------------------------
    # 2. Deflection due to panel shear 
    # --------------------------------------------------------------------
    ws.range(get_excel_range('D', panel_shear_deflection_start, no_of_storey)).options(transpose=True).value = Cumulative_force_list * 1000  # Vi, N
    ws.range(get_excel_range('E', panel_shear_deflection_start, no_of_storey)).options(transpose=True).value = shear_wall_height_list * 1000  # Hi, mm
    ws.range(get_excel_range('F', panel_shear_deflection_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list * 1000
    ws.range(get_excel_range('G', panel_shear_deflection_start, no_of_storey)).options(transpose=True).value = bv_list
    ws.range(get_excel_range('H', panel_shear_deflection_start, no_of_storey)).options(transpose=True).value = D_panel_shear_list

    # --------------------------------------------------------------------
    # --- 3. Deflection due to nail slip 
    # --------------------------------------------------------------------
    
    nail_slip_start = 234

    ws.range(get_excel_range("D", nail_slip_start, no_of_storey)).options(transpose=True).value = v_list
    ws.range(get_excel_range("E", nail_slip_start, no_of_storey)).options(transpose=True).value = nail_spacing_list
    ws.range(get_excel_range("F", nail_slip_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list * 1000
    ws.range(get_excel_range("G", nail_slip_start, no_of_storey)).options(transpose=True).value = v_s_list
    ws.range(get_excel_range("H", nail_slip_start, no_of_storey)).options(transpose=True).value = nail_diameter_list
    ws.range(get_excel_range("I", nail_slip_start, no_of_storey)).options(transpose=True).value = en_list
    ws.range(get_excel_range("J", nail_slip_start, no_of_storey)).options(transpose=True).value = D_nail_slip


    # --------------------------------------------------------------------
    # --- 4. Deflection due to vertical elongation of the wall anchorage system ---
    # --------------------------------------------------------------------
    anchorage_start = 246

    ws.range(get_excel_range("D", anchorage_start, no_of_storey)).options(transpose=True).value = Tf_list
    ws.range(get_excel_range("H", anchorage_start, no_of_storey)).options(transpose=True).value = Cf_list
    ws.range(get_excel_range("G", anchorage_start, no_of_storey)).options(transpose=True).value = seating_increments_mm  # Del_r
    ws.range(get_excel_range("F", anchorage_start, no_of_storey)).options(transpose=True).value = deflections_mm         # Del_f
    ws.range(get_excel_range("E", anchorage_start, no_of_storey)).options(transpose=True).value = tr_values
    ws.range(get_excel_range("I", anchorage_start, no_of_storey)).options(transpose=True).value = eperp_list
    ws.range(get_excel_range("J", anchorage_start, no_of_storey)).options(transpose=True).value = Ac_list
    ws.range(get_excel_range("K", anchorage_start, no_of_storey)).options(transpose=True).value = Lc_list  
    ws.range(get_excel_range("L", anchorage_start, no_of_storey)).options(transpose=True).value = d_a_list
    ws.range(get_excel_range("M", anchorage_start, no_of_storey)).options(transpose=True).value = D_wall_anchorage_list


    # --------------------------------------------------------------------
    # --- 5. Deflection due to rotation ---
    # --------------------------------------------------------------------


    rotation_start = 262

    ws.range(get_excel_range("D", rotation_start, no_of_storey)).options(transpose=True).value = shear_wall_height_list * 1000
    ws.range(get_excel_range("E", rotation_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list * 1000
    ws.range(get_excel_range("F", rotation_start, no_of_storey)).options(transpose=True).value = theta_list
    ws.range(get_excel_range("G", rotation_start, no_of_storey)).options(transpose=True).value = theta_cumsum
    ws.range(get_excel_range("H", rotation_start, no_of_storey)).options(transpose=True).value = d_a_list
    ws.range(get_excel_range("I", rotation_start, no_of_storey)).options(transpose=True).value = alpha_cumsum  
    ws.range(get_excel_range("J", rotation_start, no_of_storey)).options(transpose=True).value = D_wall_anchorage_list
    ws.range(get_excel_range("K", rotation_start, no_of_storey)).options(transpose=True).value = D_rotation_bottom_list  

    # --------------------------------------------------------------------
    # --- Total deflection summary table data ---
    # --------------------------------------------------------------------


    deflection_summary_start = 177

    ws.range(get_excel_range("D", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_bending_list
    ws.range(get_excel_range("E", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_panel_shear_list
    ws.range(get_excel_range("F", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_nail_slip
    ws.range(get_excel_range("G", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_wall_anchorage_list
    ws.range(get_excel_range("H", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_rotation_bottom_list
    ws.range(get_excel_range("I", deflection_summary_start, no_of_storey)).options(transpose=True).value = D_total_list


    # --------------------------------------------------------------------
    # --- Optionally, write the results back to Excel ---
    # --------------------------------------------------------------------


    optional_output_start = 40

    ws.range(get_excel_range("AQ", optional_output_start, no_of_storey)).options(transpose=True).value = epar_list
    ws.range(get_excel_range("AR", optional_output_start, no_of_storey)).options(transpose=True).value = eperp_list
    ws.range(get_excel_range("AT", optional_output_start, no_of_storey)).options(transpose=True).value = Ac_list  


    # --------------------------------------------------------------------
    # --- Fundamental period of the wall ---
    # --------------------------------------------------------------------

    period_start = 276

    ws.range(get_excel_range("E", period_start, no_of_storey)).options(transpose=True).value = Force_list
    ws.range(get_excel_range("F", period_start, no_of_storey)).options(transpose=True).value = D_total_list
    ws.range(get_excel_range("G", period_start, no_of_storey)).options(transpose=True).value = t_upper
    ws.range(get_excel_range("H", period_start, no_of_storey)).options(transpose=True).value = t_lower
    ws.range(get_excel_range("I", period_start, 1)).options(transpose=True).value = fundamental_period  # Single cell only


    # --------------------------------------------------------------------
    # --- Wall stiffness calculation ---
    # --------------------------------------------------------------------


    stiffness_start = 292

    ws.range(get_excel_range("D", stiffness_start, no_of_storey)).options(transpose=True).value = v_list
    ws.range(get_excel_range("E", stiffness_start, no_of_storey)).options(transpose=True).value = shear_wall_height_list * 1000
    ws.range(get_excel_range("F", stiffness_start, no_of_storey)).options(transpose=True).value = epar_list
    ws.range(get_excel_range("G", stiffness_start, no_of_storey)).options(transpose=True).value = Ac_list
    ws.range(get_excel_range("H", stiffness_start, no_of_storey)).options(transpose=True).value = shear_wall_length_list * 1000
    ws.range(get_excel_range("I", stiffness_start, no_of_storey)).options(transpose=True).value = bv_list
    ws.range(get_excel_range("J", stiffness_start, no_of_storey)).options(transpose=True).value = en_list
    ws.range(get_excel_range("K", stiffness_start, no_of_storey)).options(transpose=True).value = d_a_list
    ws.range(get_excel_range("L", stiffness_start, no_of_storey)).options(transpose=True).value = stiffness_list
    ws.range(get_excel_range("M", stiffness_start, 1)).options(transpose=True).value = average_stiffness  # Single value


    # --------------------------------------------------------------------
    # --- Copy the values used by the analysis to the next table ---
    # --------------------------------------------------------------------


    input_summary_start = 50

    ws.range(get_excel_range("C", input_summary_start, no_of_storey)).options(transpose=True).value = panel_type_list
    ws.range(get_excel_range("D", input_summary_start, no_of_storey)).options(transpose=True).value = panel_thickness_list
    ws.range(get_excel_range("E", input_summary_start, no_of_storey)).options(transpose=True).value = panel_side_list
    ws.range(get_excel_range("F", input_summary_start, no_of_storey)).options(transpose=True).value = nail_length_list
    ws.range(get_excel_range("G", input_summary_start, no_of_storey)).options(transpose=True).value = nail_diameter_list
    ws.range(get_excel_range("H", input_summary_start, no_of_storey)).options(transpose=True).value = nail_spacing_list

    ws.range(get_excel_range("I", input_summary_start, no_of_storey)).options(transpose=True).value = v_rs_govern_array

    ws.range(get_excel_range("L", input_summary_start, no_of_storey)).options(transpose=True).value = holddown_type_list
    ws.range(get_excel_range("M", input_summary_start, no_of_storey)).options(transpose=True).value = holddown_model_list
    ws.range(get_excel_range("N", input_summary_start, no_of_storey)).options(transpose=True).value = holddown_quantity_list
    ws.range(get_excel_range("O", input_summary_start, no_of_storey)).options(transpose=True).value = tr_values
    ws.range(get_excel_range("P", input_summary_start, no_of_storey)).options(transpose=True).value = Tf_list

    ws.range(get_excel_range("R", input_summary_start, no_of_storey)).options(transpose=True).value = stud_type_list
    ws.range(get_excel_range("S", input_summary_start, no_of_storey)).options(transpose=True).value = no_of_stud_list
    ws.range(get_excel_range("T", input_summary_start, no_of_storey)).options(transpose=True).value = cr_list
    ws.range(get_excel_range("U", input_summary_start, no_of_storey)).options(transpose=True).value = Cf_list


    # --------------------------------------------------------------------
    # --- ATS system component summary ---
    # --------------------------------------------------------------------

    
    ats_summary_start = 94

    ws.range(get_excel_range("D", ats_summary_start, no_of_storey)).options(transpose=True).value = holddown_type_list
    ws.range(get_excel_range("E", ats_summary_start, no_of_storey)).options(transpose=True).value = holddown_model_list
    ws.range(get_excel_range("F", ats_summary_start, no_of_storey)).options(transpose=True).value = holddown_quantity_list
    ws.range(get_excel_range("G", ats_summary_start, no_of_storey)).options(transpose=True).value = tr_values
    ws.range(get_excel_range("H", ats_summary_start, no_of_storey)).options(transpose=True).value = Tf_list

    ws.range(get_excel_range("M", ats_summary_start, no_of_storey)).options(transpose=True).value = bp_factored_resistances
    ws.range(get_excel_range("O", ats_summary_start, no_of_storey)).options(transpose=True).value = seating_increments_mm  # Del_r
    ws.range(get_excel_range("P", ats_summary_start, no_of_storey)).options(transpose=True).value = deflections_mm         # Del_f


    # ------------------------------
    # Deflection Summary Table
    # ------------------------------
    deflection_start_row = 82

    # 1. Deflection due to bending
    ws.range(get_excel_range("D", deflection_start_row, no_of_storey)).options(transpose=True).value = M_cumulative
    ws.range(get_excel_range("E", deflection_start_row, no_of_storey)).options(transpose=True).value = EI_tr_list
    ws.range(get_excel_range("F", deflection_start_row, no_of_storey)).options(transpose=True).value = D_bending_list

    # 2. Deflection due to panel shear
    ws.range(get_excel_range("G", deflection_start_row, no_of_storey)).options(transpose=True).value = bv_list
    ws.range(get_excel_range("H", deflection_start_row, no_of_storey)).options(transpose=True).value = D_panel_shear_list

    # 3. Deflection due to nail slip
    ws.range(get_excel_range("I", deflection_start_row, no_of_storey)).options(transpose=True).value = v_s_list
    ws.range(get_excel_range("J", deflection_start_row, no_of_storey)).options(transpose=True).value = en_list
    ws.range(get_excel_range("K", deflection_start_row, no_of_storey)).options(transpose=True).value = D_nail_slip

    # 4. Deflection due to wall anchorage
    ws.range(get_excel_range("L", deflection_start_row, no_of_storey)).options(transpose=True).value = seating_increments_mm
    ws.range(get_excel_range("M", deflection_start_row, no_of_storey)).options(transpose=True).value = deflections_mm
    ws.range(get_excel_range("N", deflection_start_row, no_of_storey)).options(transpose=True).value = d_a_list
    ws.range(get_excel_range("O", deflection_start_row, no_of_storey)).options(transpose=True).value = D_wall_anchorage_list

    # 5. Deflection due to rotation
    ws.range(get_excel_range("P", deflection_start_row, no_of_storey)).options(transpose=True).value = theta_cumsum
    ws.range(get_excel_range("Q", deflection_start_row, no_of_storey)).options(transpose=True).value = alpha_cumsum
    ws.range(get_excel_range("R", deflection_start_row, no_of_storey)).options(transpose=True).value = D_rotation_bottom_list

    # Total deflection and wall height
    ws.range(get_excel_range("S", deflection_start_row, no_of_storey)).options(transpose=True).value = D_total_list
    ws.range(get_excel_range("T", deflection_start_row, no_of_storey)).options(transpose=True).value = shear_wall_height_list * 1000

    # Drift ratio
    ws.range(get_excel_range("U", deflection_start_row, no_of_storey)).options(transpose=True).value = (D_total_list / (shear_wall_height_list * 1000))

    # ----------------------------------------
    # Shear clips and nail spacing data table
    # ----------------------------------------
    clip_start_row = 69

    ws.range(get_excel_range("D", clip_start_row, no_of_storey)).options(transpose=True).value = ve_list
    ws.range(get_excel_range("F", clip_start_row, no_of_storey)).options(transpose=True).value = factored_resistance_values
    ws.range(get_excel_range("G", clip_start_row, no_of_storey)).options(transpose=True).value = (np.array(factored_resistance_values) / np.array(ve_list)) * 1000


    wb.close()
