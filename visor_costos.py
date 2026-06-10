import json
import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGIN_URL = "https://api.ecomexperts.com/users/users/doLogin.json"
GRAPHQL_URL = "https://api.ecomexperts.com/graphql"
ACCOUNT_ID_OBJETIVO = "33833"
TIMEOUT = (20, 90)
PAGE_DELAY = 0.05

COMMON_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

st.set_page_config(page_title="Visor de Costos", page_icon="📦", layout="wide")


def build_session():
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(COMMON_HEADERS)
    return session


def post_graphql(session, query):
    payload = {"query": query, "operationName": None}
    resp = session.post(GRAPHQL_URL, data=json.dumps(payload), timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise ValueError(f"GraphQL error: {data['errors']}")
    return data


def login_session(email, password):
    session = build_session()
    payload_login = {"User": {"email_address": email, "password": password}}
    resp = session.post(LOGIN_URL, data=json.dumps(payload_login), timeout=TIMEOUT)
    resp.raise_for_status()

    try:
        body = resp.json()
    except Exception:
        body = {}

    if isinstance(body, dict) and (body.get("error") or body.get("errors")):
        raise ValueError(f"Login inválido: {body.get('error') or body.get('errors')}")
    if len(session.cookies) == 0:
        raise ValueError("Login inválido: no se recibió cookie de sesión.")
    return session


def fetch_ml_listings_fast(session, status_callback=None):
    all_rows = []
    current_page = 1

    while True:
        if status_callback:
            status_callback(f"Consultando mlListings - página {current_page}...")

        query = f"""
        query {{
          mlListings {{
            find(page: {current_page}, filters:[{{ filter:"active", values:["1"] }}]) {{
              data {{
                id
                accountId
                owner
                ownerId
                productListings {{
                  qty
                  productId
                  productVariantId
                  product {{
                    sku
                    title
                  }}
                }}
              }}
            }}
          }}
        }}
        """

        data = post_graphql(session, query)
        listings = data.get("data", {}).get("mlListings", {}).get("find", {}).get("data", [])
        if not listings:
            break

        for listing in listings:
            account_id = str(listing.get("accountId", ""))
            if account_id != ACCOUNT_ID_OBJETIVO:
                continue

            owner = str(listing.get("owner", ""))
            mla = str(listing.get("ownerId", ""))
            for item in listing.get("productListings") or []:
                product = item.get("product") or {}
                all_rows.append(
                    {
                        "mla": mla,
                        "owner": owner,
                        "account_id": account_id,
                        "product_id": str(item.get("productId", "") or ""),
                        "product_variant_id": str(item.get("productVariantId", "") or ""),
                        "sku": str(product.get("sku", "") or ""),
                        "titulo_producto_base": str(product.get("title", "") or ""),
                        "unidades": pd.to_numeric(item.get("qty"), errors="coerce"),
                    }
                )

        current_page += 1
        if PAGE_DELAY:
            time.sleep(PAGE_DELAY)

    return pd.DataFrame(all_rows)


def fetch_products_fast(session, status_callback=None):
    rows = []
    current_page = 1

    while True:
        if status_callback:
            status_callback(f"Consultando products - página {current_page}...")

        query = f"""
        query {{
          products {{
            find(page: {current_page}) {{
              data {{
                sku
                title
                tax
                variants {{
                  sku
                  cost
                }}
              }}
            }}
          }}
        }}
        """

        data = post_graphql(session, query)
        products = data.get("data", {}).get("products", {}).get("find", {}).get("data", [])
        if not products:
            break

        for product in products:
            product_sku = str(product.get("sku", "") or "")
            product_title = str(product.get("title", "") or "")
            product_tax = product.get("tax", None)

            tax_value = None
            if isinstance(product_tax, dict):
                for k in ["iva", "IVA", "tax", "value", "amount", "percentage", "percent"]:
                    if k in product_tax:
                        tax_value = product_tax[k]
                        break
            else:
                tax_value = product_tax

            variants = product.get("variants") or []
            if variants:
                costs = [pd.to_numeric(v.get("cost"), errors="coerce") for v in variants]
                costs = [c for c in costs if pd.notna(c)]
                max_cost = max(costs) if costs else None
            else:
                max_cost = None

            rows.append(
                {
                    "sku": product_sku,
                    "titulo_catalogo": product_title,
                    "costo_unitario": pd.to_numeric(max_cost, errors="coerce"),
                    "iva": pd.to_numeric(tax_value, errors="coerce"),
                }
            )

        current_page += 1
        if PAGE_DELAY:
            time.sleep(PAGE_DELAY)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sku", "titulo_catalogo", "costo_unitario", "iva"])

    df["sku"] = df["sku"].astype(str)
    return (
        df.groupby("sku", as_index=False)
        .agg(
            {
                "titulo_catalogo": "first",
                "costo_unitario": "max",
                "iva": "max",
            }
        )
    )


def clasificar_mla(num_skus, total_qty):
    if num_skus == 1 and total_qty == 1:
        return "monoproducto"
    if num_skus == 1 and total_qty > 1:
        return "monoproducto multioferta"
    if num_skus > 1:
        return "combo"
    return "sin clasificar"


def build_outputs(df_listings, df_products, status_callback=None):
    if df_listings.empty:
        detalle = pd.DataFrame(
            columns=[
                "mla",
                "sku",
                "unidades",
                "costo_unitario",
                "costo_total_sku",
                "iva",
                "titulo_final",
            ]
        )
        final_df = pd.DataFrame(
            columns=[
                "mla",
                "titulo_producto",
                "skus_asociados",
                "unidades_totales",
                "costo_total_mla",
                "iva",
                "cant_sku",
                "tipo_producto",
            ]
        )
        return detalle, final_df

    if status_callback:
        status_callback("Agrupando detalle por MLA + SKU...")

    detalle = (
        df_listings.groupby(["mla", "sku"], as_index=False)
        .agg(
            {
                "unidades": "sum",
                "titulo_producto_base": lambda s: " | ".join(
                    sorted(set([str(x) for x in s if str(x).strip()]))
                ),
                "owner": "first",
                "account_id": "first",
            }
        )
    )

    if status_callback:
        status_callback("Cruzando con catálogo de productos...")

    detalle = detalle.merge(df_products, on="sku", how="left")
    detalle["costo_unitario"] = pd.to_numeric(detalle["costo_unitario"], errors="coerce").fillna(0)
    detalle["iva"] = pd.to_numeric(detalle["iva"], errors="coerce")
    detalle["unidades"] = pd.to_numeric(detalle["unidades"], errors="coerce").fillna(0)
    detalle["titulo_final"] = (
        detalle["titulo_producto_base"].replace("", pd.NA).fillna(detalle["titulo_catalogo"])
    )
    detalle["costo_total_sku"] = detalle["costo_unitario"] * detalle["unidades"]

    detalle = (
        detalle[
            [
                "mla",
                "sku",
                "unidades",
                "costo_unitario",
                "costo_total_sku",
                "iva",
                "titulo_final",
            ]
        ]
        .sort_values(["mla", "sku"])
        .reset_index(drop=True)
    )

    if status_callback:
        status_callback("Construyendo resumen final por MLA...")

    final_df = (
        detalle.groupby("mla", as_index=False)
        .agg(
            titulo_producto=(
                "titulo_final",
                lambda s: " | ".join(sorted(set([str(x) for x in s if str(x).strip()]))),
            ),
            skus_asociados=(
                "sku",
                lambda s: " | ".join(sorted(set([str(x) for x in s if str(x).strip()]))),
            ),
            unidades_totales=("unidades", "sum"),
            costo_total_mla=("costo_total_sku", "sum"),
            iva=("iva", "max"),
            cant_sku=("sku", "nunique"),
        )
    )

    final_df["tipo_producto"] = final_df.apply(
        lambda r: clasificar_mla(r["cant_sku"], r["unidades_totales"]), axis=1
    )
    final_df = (
        final_df[
            [
                "mla",
                "titulo_producto",
                "skus_asociados",
                "unidades_totales",
                "costo_total_mla",
                "iva",
                "cant_sku",
                "tipo_producto",
            ]
        ]
        .sort_values(["tipo_producto", "mla"])
        .reset_index(drop=True)
    )

    return detalle, final_df


def init_state():
    defaults = {
        "authenticated": False,
        "session": None,
        "df_final": pd.DataFrame(),
        "df_detalle": pd.DataFrame(),
        "df_vista": pd.DataFrame(),
        "df_detalle_vista": pd.DataFrame(),
        "status": "Listo para consultar",
        "detalle_carga": "Sin consultas ejecutadas",
        "buscar_titulo": "",
        "buscar_sku": "",
        "buscar_mla": "",
        "tipo": "Todos",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_filters():
    if st.session_state.df_final.empty:
        st.session_state.df_vista = st.session_state.df_final.copy()
        st.session_state.df_detalle_vista = st.session_state.df_detalle.copy()
        return

    vista = st.session_state.df_final.copy()
    ft = st.session_state.buscar_titulo.strip()
    fs = st.session_state.buscar_sku.strip()
    fm = st.session_state.buscar_mla.strip()
    tp = st.session_state.tipo.strip()

    if ft:
        vista = vista[vista["titulo_producto"].astype(str).str.contains(ft, case=False, na=False)]
    if fs:
        vista = vista[vista["skus_asociados"].astype(str).str.contains(fs, case=False, na=False)]
    if fm:
        vista = vista[vista["mla"].astype(str).str.contains(fm, case=False, na=False)]
    if tp and tp != "Todos":
        vista = vista[vista["tipo_producto"] == tp]

    st.session_state.df_vista = vista.copy()

    detalle_vista = st.session_state.df_detalle.copy()
    if fm:
        detalle_vista = detalle_vista[
            detalle_vista["mla"].astype(str).str.contains(fm, case=False, na=False)
        ]
    if fs:
        detalle_vista = detalle_vista[
            detalle_vista["sku"].astype(str).str.contains(fs, case=False, na=False)
        ]
    if ft:
        detalle_vista = detalle_vista[
            detalle_vista["titulo_final"].astype(str).str.contains(ft, case=False, na=False)
        ]
    if tp and tp != "Todos":
        mlas_validos = set(vista["mla"].astype(str).tolist())
        detalle_vista = detalle_vista[
            detalle_vista["mla"].astype(str).isin(mlas_validos)
        ]

    st.session_state.df_detalle_vista = detalle_vista.copy()


def excel_bytes(df_vista, df_detalle_vista):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_vista.to_excel(writer, index=False, sheet_name="tabla_final")
        df_detalle_vista.to_excel(writer, index=False, sheet_name="detalle_mla_sku")
    return output.getvalue()


def update_status(msg: str):
    st.session_state.detalle_carga = msg


init_state()

st.title("Visor de Costos")
st.caption("Consulta costos por MLA y detalle SKU desde EcomExperts.")

# SIDEBAR LOGIN
with st.sidebar:
    st.header("Acceso")
    email = st.text_input("Correo", key="email")
    password = st.text_input("Contraseña", type="password", key="password")

    login_col1, login_col2 = st.columns(2)
    with login_col1:
        login_clicked = st.button("Ingresar", use_container_width=True)
    with login_col2:
        logout_clicked = st.button("Salir", use_container_width=True)

    if logout_clicked:
        st.session_state.authenticated = False
        st.session_state.session = None
        st.session_state.df_final = pd.DataFrame()
        st.session_state.df_detalle = pd.DataFrame()
        st.session_state.df_vista = pd.DataFrame()
        st.session_state.df_detalle_vista = pd.DataFrame()
        st.session_state.status = "Sesión cerrada"
        st.session_state.detalle_carga = "Sin consultas ejecutadas"
        st.rerun()

    if login_clicked:
        if not email or not password:
            st.warning("Debes ingresar correo y contraseña.")
        else:
            try:
                with st.spinner("Validando credenciales..."):
                    session = login_session(email, password)
                st.session_state.session = session
                st.session_state.authenticated = True
                st.success("Acceso concedido")
            except Exception as e:
                st.session_state.authenticated = False
                st.session_state.session = None
                st.error(f"No fue posible validar las credenciales: {e}")

    st.divider()
    st.write(f"Estado: {'Autenticado' if st.session_state.authenticated else 'No autenticado'}")

if not st.session_state.authenticated:
    st.info("Ingresa tus credenciales en la barra lateral para comenzar.")
    st.stop()

# BLOQUE CONSULTA
col1, col2 = st.columns([1, 3])
with col1:
    consultar = st.button("Consultar datos", type="primary", use_container_width=True)

    if consultar:
        if st.session_state.session is None:
            st.error("La sesión no es válida. Vuelve a iniciar sesión.")
        else:
            try:
                with st.spinner("Consultando información (datos frescos)..."):
                    t0 = time.time()
                    df_listings = fetch_ml_listings_fast(st.session_state.session, update_status)
                    update_status("Consultando catálogo de productos...")
                    df_products = fetch_products_fast(st.session_state.session, update_status)
                    detalle_df, final_df = build_outputs(
                        df_listings, df_products, update_status
                    )

                    st.session_state.df_detalle = detalle_df.copy()
                    st.session_state.df_final = final_df.copy()
                    st.session_state.status = (
                        f"Consulta completada en {round(time.time() - t0, 2)} s"
                    )
                    st.session_state.detalle_carga = "Carga finalizada correctamente."
                    st.session_state.buscar_titulo = ""
                    st.session_state.buscar_sku = ""
                    st.session_state.buscar_mla = ""
                    st.session_state.tipo = "Todos"
                    apply_filters()

                st.success(st.session_state.status)
            except Exception as e:
                st.session_state.status = "Error en consulta"
                st.session_state.detalle_carga = "La carga se interrumpió por un error."
                st.error(f"Error: {e}")

with col2:
    st.write(f"**Estado de carga:** {st.session_state.detalle_carga}")
    st.write(f"**Estado general:** {st.session_state.status}")

# FILTROS
st.subheader("Filtros")
filter_cols = st.columns([2, 2, 2, 2, 1])
with filter_cols[0]:
    st.text_input("Título", key="buscar_titulo", on_change=apply_filters)
with filter_cols[1]:
    st.text_input("SKU", key="buscar_sku", on_change=apply_filters)
with filter_cols[2]:
    st.text_input("MLA", key="buscar_mla", on_change=apply_filters)
with filter_cols[3]:
    tipos = ["Todos"]
    if not st.session_state.df_final.empty:
        tipos += sorted(
            st.session_state.df_final["tipo_producto"].dropna().unique().tolist()
        )
    st.selectbox("Tipo", options=tipos, key="tipo", on_change=apply_filters)
with filter_cols[4]:
    if st.button("Limpiar", use_container_width=True):
        st.session_state.buscar_titulo = ""
        st.session_state.buscar_sku = ""
        st.session_state.buscar_mla = ""
        st.session_state.tipo = "Todos"
        apply_filters()
        st.rerun()

apply_filters()

# MÉTRICAS
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("MLA filtrados", len(st.session_state.df_vista))
with m2:
    st.metric(
        "Costo total filtrado",
        round(
            st.session_state.df_vista["costo_total_mla"].sum(), 2
        )
        if not st.session_state.df_vista.empty
        else 0,
    )
with m3:
    st.metric(
        "Unidades totales filtradas",
        round(
            st.session_state.df_vista["unidades_totales"].sum(), 2
        )
        if not st.session_state.df_vista.empty
        else 0,
    )

# TABLAS
st.subheader("Tabla final por MLA")
st.dataframe(st.session_state.df_vista, use_container_width=True, hide_index=True)

st.subheader("Detalle MLA + SKU")
st.dataframe(st.session_state.df_detalle_vista, use_container_width=True, hide_index=True)

# DESCARGAS
if not st.session_state.df_vista.empty:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Descargar CSV",
            data=st.session_state.df_vista.to_csv(index=False).encode("utf-8-sig"),
            file_name="visor_costos_filtrado.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        excel_data = excel_bytes(
            st.session_state.df_vista, st.session_state.df_detalle_vista
        )
        st.download_button(
            "Descargar Excel",
            data=excel_data,
            file_name="visor_costos_filtrado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
