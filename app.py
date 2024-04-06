from flask import Flask, render_template, request
import torch
import torch.optim as optim
import plotly.graph_objs as go
import time

app = Flask(__name__)

# Fonction pour calculer D_AB
def calculate_D_AB(params, Xa, Xb, lambda_a, lambda_b, T, q_a, q_b, D_BA0, D_AB0):
    a_AB, a_BA = params

    # Convertir les valeurs d'entrée en tenseurs
    Xa_tensor = torch.tensor(Xa)
    Xb_tensor = torch.tensor(Xb)
    D_BA0_tensor = torch.tensor(D_BA0)
    D_AB0_tensor = torch.tensor(D_AB0)
    lambda_a_tensor = torch.tensor(lambda_a)
    lambda_b_tensor = torch.tensor(lambda_b)
    q_a_tensor = torch.tensor(q_a)
    q_b_tensor = torch.tensor(q_b)
    T_tensor = torch.tensor(T)

    # Calcul des termes de l'équation
    D = Xa_tensor*(D_BA0_tensor) + Xb_tensor*torch.log(D_AB0_tensor) + 2*(Xa_tensor*torch.log(Xa_tensor+(Xb_tensor*lambda_b_tensor)/lambda_a_tensor)+Xb_tensor*torch.log(Xb_tensor+(Xa_tensor*lambda_a_tensor)/lambda_b_tensor)) + 2*Xa_tensor*Xb_tensor*((lambda_a_tensor/(Xa_tensor*lambda_a_tensor+Xb_tensor*lambda_b_tensor))*(1-(lambda_a_tensor/lambda_b_tensor)) + (lambda_b_tensor/(Xa_tensor*lambda_a_tensor+Xb_tensor*lambda_b_tensor))*(1-(lambda_b_tensor/lambda_a_tensor))) + Xb_tensor*q_a_tensor*((1-((Xb_tensor*q_b_tensor*torch.exp(-a_BA/T_tensor))/(Xa_tensor*q_a_tensor+Xb_tensor*q_b_tensor*torch.exp(-a_BA/T_tensor)))**2)*(-a_BA/T_tensor)+(1-((Xb_tensor*q_b_tensor)/(Xb_tensor*q_b_tensor+Xa_tensor*q_a_tensor*torch.exp(-a_AB/T_tensor)))**2)*torch.exp(-a_AB/T_tensor)*(-a_AB/T_tensor)) + Xa_tensor*q_b_tensor*((1-((Xa_tensor*q_a_tensor*torch.exp(-a_AB/T_tensor))/(Xa_tensor*q_a_tensor*torch.exp(-a_AB/T_tensor)+Xb_tensor*q_b_tensor))**2)*(-a_AB/T_tensor)+(1-((Xa_tensor*q_a_tensor)/(Xa_tensor*q_a_tensor+Xb_tensor*q_b_tensor*torch.exp(-a_BA/T_tensor)))**2)*torch.exp(-a_BA/T_tensor)*(-a_BA/T_tensor))
    # Calcul de D_AB
    D_AB = torch.exp(D)

    return D_AB

# Fonction objectif pour la minimisation
def objective(params, D_AB_exp, Xa, Xb, lambda_a, lambda_b, T, q_a, q_b, D_BA0, D_AB0):
    D_AB_calculated = calculate_D_AB(params, Xa, Xb, lambda_a, lambda_b, T, q_a, q_b, D_BA0, D_AB0)
    return torch.abs(D_AB_calculated - D_AB_exp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Obtenir les données du formulaire
    D_AB_exp = float(request.form['D_AB_exp'])
    D_BA_exp = float(request.form['D_BA_exp'])
    T = float(request.form['T'])
    Xa = float(request.form['Xa'])
    Xb = float(request.form['Xb'])
    lambda_a = float(request.form['lambda_a'])
    lambda_b = float(request.form['lambda_b'])
    q_a = float(request.form['q_a'])
    q_b = float(request.form['q_b'])
    
    # Afficher les entrées reçues
    print("Inputs received:")
    print("D_AB_exp:", D_AB_exp)
    print("D_BA_exp:", D_BA_exp)
    print("T:", T)
    print("Xa:", Xa)
    print("Xb:", Xb)
    print("lambda_a:", lambda_a)
    print("lambda_b:", lambda_b)
    print("q_a:", q_a)
    print("q_b:", q_b)

    # Paramètres initiaux
    params_initial = torch.tensor([600.0, 700.0], requires_grad=True)

    # Optimiseur (utilisation de Adam avec un taux d'apprentissage plus bas)
    optimizer = optim.Adam([params_initial], lr=1)

    # Nombre maximal d'itérations
    iteration = 0

    # Début du chronomètre
    start_time = time.time()

    # Boucle d'ajustement des paramètres
    while True:
        # Réinitialiser les gradients
        optimizer.zero_grad()
        # Calcul de l'objectif
        loss = objective(params_initial, D_AB_exp, Xa, Xb, lambda_a, lambda_b, T, q_a, q_b, D_BA_exp, D_AB_exp)
        # Rétropropagation
        loss.backward()
        # Mise à jour des paramètres
        optimizer.step()

        # Paramètres optimisés
        a_AB_opt, a_BA_opt = params_initial.detach()

        # Calcul de D_AB avec les paramètres optimisés
        D_AB_opt = calculate_D_AB([a_AB_opt, a_BA_opt], Xa, Xb, lambda_a, lambda_b, T, q_a, q_b, D_BA_exp, D_AB_exp).item()

        # Convertir D_AB_opt et D_AB_exp en tenseurs
        D_AB_opt_tensor = torch.tensor(D_AB_opt)
        D_AB_exp_tensor = torch.tensor(D_AB_exp)

        # Calcul de l'erreur entre D_AB_exp et D_AB_opt
        error = torch.abs(D_AB_opt_tensor - D_AB_exp_tensor)
        # Vérification de convergence
        if error <= 1e-10:
            print("Convergence achieved!")
            break

        # Incrémentation du nombre d'itérations
        iteration += 1

    # Fin du chronomètre
    end_time = time.time()

    # Calcul de la durée de l'itération
    iteration_duration = end_time - start_time
    print("Duration of iteration:", iteration_duration, "seconds")

    # Liste des valeurs de Xa pour lesquelles nous voulons tracer D_AB
    Xa_values = torch.linspace(0, 0.7, 100)

    # Calcul de D_AB pour chaque valeur de Xa
    D_AB_values = [calculate_D_AB([a_AB_opt, a_BA_opt], Xa_val, 1 - Xa_val, lambda_a, lambda_b, T, q_a, q_b, D_BA_exp, D_AB_exp).item() for Xa_val in Xa_values]
    
    # Tracer la variation du coefficient de diffusion en fonction de Xa avec Plotly
    trace = go.Scatter(x=Xa_values.numpy(), y=D_AB_values, mode='lines', name='D_AB en fonction de Xa')
    plot_data = [trace]
    plot_layout = go.Layout(title='Variation du coefficient de diffusion en fonction de la fraction molaire Xa',
                            xaxis=dict(title='Fraction molaire Xa'),
                            yaxis=dict(title='Coefficient de diffusion D_AB (cm^2/s)'),
                            legend=dict(x=0, y=1),
                             width=800,  
                            height=700,
                            plot_bgcolor='rgba(10,0,5,0)'
                           )
    plot_fig = go.Figure(data=plot_data, layout=plot_layout)
    plot_div = plot_fig.to_html(full_html=False)

    # Affichage des résultats finaux
    a_AB_str = f'a_AB = {a_AB_opt}'
    a_BA_str = f'a_BA = {a_BA_opt}'
    Dapp_str = f'Dapp: {D_AB_opt}'
    error_str = f'ERROR: {error}'

    # Calcul de la durée de l'itération en millisecondes
    iteration_duration_ms = iteration_duration * 1000

    # Rendu du modèle avec les résultats du calcul et le graphique Plotly
    return render_template('output.html', a_AB=a_AB_str, a_BA=a_BA_str, Dapp=Dapp_str, error=error_str, plot_div=plot_div, iterations=iteration, iteration_time=iteration_duration_ms)

if __name__ == '__main__':
    app.run(debug=False)
