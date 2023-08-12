#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


# In[2]:


lead=pd.read_csv("Leads.csv")
lead.head()


# In[3]:


lead.shape


# In[4]:


lead.info()


# In[5]:


lead.describe()


# If we are looking the data we see there are missing value in our data

# # Data Cleaning

# In our data there is Select value becouse customeres are not selected anything so its showing the select we need to convert select to null value

# In[6]:


lead=lead.replace("Select",np.nan)


# In[7]:


(lead.isnull().sum()/lead.shape[0]*100).sort_values(ascending=False)


# There are some columns is high percentage of missing value so we need to drop more than 40% missing value

# In[8]:


lead=lead.drop(columns=['How did you hear about X Education','Lead Profile','Lead Quality','Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Activity Index','Asymmetrique Profile Index'])


# In[9]:


(lead.isnull().sum()/lead.shape[0]*100).sort_values(ascending=False)


# # 1.City

# In[10]:


plt.figure(figsize=(10,5))

sns.countplot(lead["City"])
plt.xticks(rotation=90)


# We can see in this chart the highest peaople are in Mumbai.. so we can impute missing value with Mumbai

# In[11]:


lead['City']=lead['City'].replace(np.nan,'Mumbai')


# In[12]:


round(lead['City'].value_counts(normalize=True),2)*100


# # 2.Specialization

# 37% Missing Value

# In[88]:


plt.figure(figsize=(10,5))
sns.countplot(lead['Specialization'])
plt.xticks(rotation=90)


# In[14]:


lead['Specialization']=lead['Specialization'].replace(np.nan,'Other')


# In[15]:


round(lead['Specialization'].value_counts(normalize=True),2)*100


# # 3.Tags

# 37% missing value

# In[16]:


plt.figure(figsize=(10,5))
sns.countplot(lead['Tags'])

plt.xticks(rotation=90)


# In[17]:


lead['Tags']=lead['Tags'].replace(np.nan,'Will revert after reading the email')


# # 4.What matters most to you in choosing a course

# 30% Missing value

# In[18]:


sns.countplot(lead['What matters most to you in choosing a course'])
plt.xticks(rotation=45)


# In[19]:


lead=lead.drop('What matters most to you in choosing a course',axis=1)


# # 5. 'What is your current occupation'¶
# this column has 29% missing values

# In[20]:


sns.countplot(lead['What is your current occupation'])
plt.xticks(rotation=45)


# In[21]:


# Imputing the missing data in the 'What is your current occupation' column with 'Unemployed'
lead['What is your current occupation']=lead['What is your current occupation'].replace(np.nan,'Unemployed')


# # 6.Country

# This column has 27% missing values

# In[22]:


plt.figure(figsize=(17,5))
sns.countplot(lead['Country'])
plt.xticks(rotation=90)


# We can see that this is highly skewed column but it is an important information w.r.t. to the lead. Since most values are 'India' , we can impute missing values in this column with this value.

# In[23]:


# Imputing the missing data in the 'Country' column with 'India'
lead['Country']=lead['Country'].replace(np.nan,'India')


# In[24]:


(lead.isnull().sum()/lead.shape[0]*100).sort_values(ascending=False)


# In[25]:


# Dropping the rows with null values
lead.dropna(inplace = True)


# In[26]:


(lead.isnull().sum()/lead.shape[0]*100).sort_values(ascending=False)


# Now we don't have any missing value in the datase

# # Univariate Analysis and Bivariate Analysis

# Converted is the Target variable 

# In[27]:


lead[lead.duplicated()]


# # 1.Converted

# In[28]:


Converted = (sum(lead['Converted'])/len(lead['Converted'].index))*100
Converted


# # 2.Lead Origin

# In[29]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Origin", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 45)

Lead origin- 1.API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are                         considerable.
             2.Lead Add Form has more than 90% conversion rate but count of lead are not very high.
             3.Lead Import are very less in count.
# # 3.Lead Source

# In[30]:


plt.figure(figsize=(10,5))
sns.countplot(x='Lead Source',hue="Converted", data = lead,palette='Set1')
plt.xticks(rotation =90)


# In[31]:


# Need to replace 'google' with 'Google'
lead['Lead Source'] = lead['Lead Source'].replace(['google'], 'Google')


# In[32]:


# Creating a new category 'Others' for some of the Lead Sources which do not have much values.
lead['Lead Source'] = lead['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


# In[33]:


plt.figure(figsize=(10,5))
sns.countplot(x='Lead Source',hue="Converted", data = lead,palette='Set1')
plt.xticks(rotation =90)


# # Inference
# Google and Direct traffic generates maximum number of leads.
# Conversion Rate of reference leads and leads through welingak website is high.
# To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# # 4.Do Not Email

# In[34]:


plt.figure(figsize=(10,5))
sns.countplot(x='Do Not Email',hue="Converted", data = lead,palette='Set1')
plt.xticks(rotation =90)


# Most entry are NO .

# # 5.Do Not Call

# In[35]:


sns.countplot(x='Do Not Call',hue="Converted", data = lead,palette='Set1')
plt.xticks(rotation =90)


# Most Entries are NO.

# # 6.TotalVisits

# In[36]:


lead['TotalVisits'].describe(percentiles=[0.05,.25,.5,.75,.90,.95,.99])


# In[37]:


sns.boxplot(lead['TotalVisits'])


# As we can see there are a number of outliers in the data. We will cap the outliers to 95% value for analysis.

# In[38]:


percentiles = lead['TotalVisits'].quantile([0.05,0.95]).values
lead['TotalVisits'][lead['TotalVisits'] <= percentiles[0]] = percentiles[0]
lead['TotalVisits'][lead['TotalVisits'] >= percentiles[1]] = percentiles[1]


# In[39]:


sns.boxplot(lead['TotalVisits'])


# In[40]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead,palette='Set1')


# Median for not converted and converted is same 

# # 7.Total Time Spent on Website

# In[41]:


lead['Total Time Spent on Website'].describe()


# In[42]:


sns.boxplot(lead['Total Time Spent on Website'])


# In[43]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead,palette='Set1')


# Leads time spend with website is more likly to converted

# # 8.Page Views Per Visit

# In[44]:


lead['Page Views Per Visit'].describe()


# In[45]:


sns.boxplot(lead['Page Views Per Visit'],orient='vert',palette='Set1')


# In[46]:


percentiles = lead['Page Views Per Visit'].quantile([0.05,0.95]).values
lead['Page Views Per Visit'][lead['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
lead['Page Views Per Visit'][lead['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[47]:


sns.boxplot(lead['Page Views Per Visit'],orient='vert',palette='Set1')


# In[48]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data =lead,palette='Set1')


# Median is same for both converted and non-converted

# # 9.Last Activity

# In[49]:


lead['Last Activity'].describe()


# In[50]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# In[51]:


# We can club the last activities to "Other_Activity" which are having less data.
lead['Last Activity'] = lead['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


# In[52]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# 1)Most of the lead have their Email opened as their last activity.
# 2)Conversion rate for leads with last activity as SMS Sent is almost 60%.

# # 10.Country

# In[53]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Country", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# No reference all are belongs to India so 

# # 11.Specialization

# In[54]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Specialization", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Focus should be more in specialization high conversational rate 

# # 12.What is your current occupation

# In[55]:


plt.figure(figsize=(15,6))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Woking professional more likely to convert the leads 
# Unemployed conversional rate 30-35%

# # 13.Search

# In[56]:


plt.figure(figsize=(10,5))
sns.countplot(x='Search', hue='Converted',data=lead)


# Most of are No reference 

# # 14.Magazine

# In[57]:


sns.countplot(x = "Magazine", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# In[59]:


lead.columns


# # 15.Newspaper Article

# In[60]:


sns.countplot(x = "Newspaper Article", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 16.X Education Forums

# In[61]:


sns.countplot(x = "X Education Forums", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 17.Newspaper

# In[62]:


sns.countplot(x = "Newspaper", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 18.Digital Advertisement

# In[63]:


sns.countplot(x = "Digital Advertisement", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 19.Through Recommendations

# In[64]:


sns.countplot(x = "Through Recommendations", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 20.Receive More Updates About Our Courses

# In[65]:


sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.
# 

# # 21.Tags

# In[66]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Tags", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# # 22.Update me on Supply Chain Content

# In[67]:


sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 23.Get updates on DM Content

# In[68]:


sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 24.City

# In[69]:


plt.figure(figsize=(15,5))
sns.countplot(x = "City", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most of the people are from Mumbai

# # 25.I agree to pay the amount through cheque

# In[70]:


sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 26.A free copy of Mastering The Interview

# In[71]:


sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# Most entries are 'No'. No Inference can be drawn with this parameter.

# # 27.Last Notable Activity

# In[72]:


plt.figure(figsize=(15,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead,palette='Set1')
plt.xticks(rotation = 90)


# In[74]:


lead_data = lead.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article','X Education Forums',
                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                            'A free copy of Mastering The Interview'],1)


# In[76]:


lead_data.columns


# In[77]:


lead_data.info()


# # Data Preparation

# 1.Converting Binary (YES/NO) To (1/0)

# In[78]:


vars =  ['Do Not Email', 'Do Not Call']

def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

lead_data[vars] = lead_data[vars].apply(binary_map)


# 2.Creating Dummy Variables

# In[86]:


dummy_data = pd.get_dummies(lead_data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity']], drop_first=True)
dummy_data.head()


# In[ ]:




