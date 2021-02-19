# #--- setup with credentials
# mkdir -p ~/.streamlit/

# echo "\
# [general]\n\
# email = \"your-email@domain.com\"\n\
# " > ~/.streamlit/credentials.toml

# echo "\
# [server]\n\
# headless = true\n\
# enableCORS=false\n\
# port = $PORT\n\
# " > ~/.streamlit/config.toml

#-- setup without crednetials
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
