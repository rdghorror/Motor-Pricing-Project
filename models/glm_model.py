import statsmodels.api as sm
import statsmodels.formula.api as smf

def train_glm(X_train, y_train, family="gamma"):
    X_train = sm.add_constant(X_train)  # Add intercept
    
    # Select the appropriate GLM family
    if family == "gamma":
        glm_family = sm.families.Gamma(sm.families.links.log())
    elif family == "tweedie":
        glm_family = sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
    else:
        raise ValueError("Unsupported family. Choose 'gamma' or 'tweedie'.")

    glm_model = sm.GLM(y_train, X_train, family=glm_family)
    glm_result = glm_model.fit()
    return glm_result

def predict_glm(glm_model, X_test):
    X_test = sm.add_constant(X_test)  # Add intercept
    return glm_model.predict(X_test)
