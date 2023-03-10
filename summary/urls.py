from django.urls import path, include
from summary import pages, logged_pages, admin_pages

urlpatterns = [

    path('login', pages.login, name='login'),
    path('logout', pages.log_out, name='log_out'),
    path('summary_dash', logged_pages.dashboard, name='dashboard'),
    path('adv_summary/<int:summary_id>',
         logged_pages.adv_summary, name='advance_summary'),
    path('nw_newsarticle', logged_pages.nw_newsarticle, name='frm_URL'),
    path('new_summary', logged_pages.new_summary, name='fromFILE'),
    path('nw_mic', logged_pages.nw_mic, name='fromAUdioRecord'),

    path('nw_own_article', logged_pages.nw_own_article, name='newAritcle'),

    path('sendApproval/<str:type>/<int:id>',
         logged_pages.sendApproval, name='sendApproval'),

    path('mysummaries',
         logged_pages.mySummaries, name='mySummaries'),

    path('viewSummary/<int:summary_id>',
         logged_pages.viewSummary, name='viewSummary'),
    path('adv_view_summary/<int:summary_id>',
         logged_pages.adv_view_summary, name='advance_view_summary'),

    path('translatePage/<str:type>/<int:summary_id>',
         logged_pages.translatePage, name='Translate page'),

    path('translate',
         logged_pages.translatefunc, name='Translate'),

    path('my_translates',
         logged_pages.my_translates, name='My translates'),
    path('viewTrans/<int:transid>',
         logged_pages.viewTrans, name='viewTrans'),
    path('new_content',
         logged_pages.new_content, name='newConetnt'),
    path('new_content_gen',
         logged_pages.new_content_gen, name='new_content_gen_text'),
    #     path('new_content_gen_img',
    #          logged_pages.new_content_gen_img, name='new_content_gen_img'),

    path('generated_content/<int:content_id>',
         logged_pages.generated_content, name='new_content_gen_img'),

    path('new_content_history',
         logged_pages.new_content_history, name='new_content_history'),


    path('twitter_analysis',
         logged_pages.twitter_analysis, name='twitter_analysis'),

    path('twitter_analysis_reports',
         logged_pages.twitter_analysis_reports, name='twitter_analysis_reports'),

    path('analysis_report/<int:analysis_id>',
         logged_pages.analysis_report, name='analysis_report'),

    path('postMeeting',
         logged_pages.postMeeting, name='postMeeting'),

    path('mymeetings',
         logged_pages.mymeetings, name='mymeetings'),

    path('view_meeting_gen/<int:meeting_id>',
         logged_pages.view_meeting_gen, name='view_meeting_gen'),




    path('post_linkedin', logged_pages.send_to_LinkedIn, name='Send LinkedIn'),
    path('post_twitter', logged_pages.send_to_twitter, name='SendTweet'),
    path('post_insta', logged_pages.underDev, name='UnderDev'),
    path('go', logged_pages.go, name='GO'),
    path('gen_image', logged_pages.gen_image, name='gen_image'),





    #     ADMIN MODULES
    path('admin_dash', admin_pages.dashboard, name='Admin Dashboard'),
    path('pending_req', admin_pages.pending_requests, name='Pending Requests'),
    path('approve_req/<int:req_id>',
         admin_pages.admin_approve_request, name='Approve Requests'),
    path('reject_req/<int:req_id>',
         admin_pages.admin_reject_request, name='Reject Requests'),
    path('admin_view_user_list',
         admin_pages.admin_view_user_list, name='View list user'),
    path('admin_view_user/<int:user_id>',
         admin_pages.admin_view_user, name='View Single user'),
    path('admin_view_single_request/<int:req_id>',
         admin_pages.admin_view_single_request, name='View Single Request'),
    path('admin_mem_list', admin_pages.admin_mem_list, name='View UserList'),






    # APIS
    # TEST_API



]
